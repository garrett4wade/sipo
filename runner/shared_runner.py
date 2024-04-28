from collections import defaultdict
import logging
import multiprocessing as mp
import numpy as np
import os
import queue
import time
import torch
import wandb

from algorithm.trainers.rkhspo import RKHSPO
from algorithm.policy import RolloutRequest, RolloutResult
from algorithm.trainer import SampleBatch
from algorithm.trainer import make as make_trainer
from algorithm.modules import gae_trace, masked_normalization, rspo_gae_trace
from utils.namedarray import recursive_apply, array_like, recursive_aggregate
from utils.timing import Timing

logger = logging.getLogger('shared_runner')
logger.setLevel(logging.INFO)


class SharedRunner:

    def __init__(self,
                 iteration,
                 all_args,
                 policy,
                 archive_policies,
                 archive_trajs,
                 storages,
                 env_ctrls,
                 info_queue,
                 eval_storages,
                 eval_env_ctrls,
                 eval_info_queue,
                 device,
                 run_dir=None):

        self.iteration = iteration
        self.all_args = all_args
        self.device = device
        self.num_agents = all_args.num_agents

        self.warm_up_rate = all_args.warm_up_rate
        self.archive_policies = archive_policies
        self.archive_trajs = archive_trajs

        # parameters
        self.env_name = self.all_args.env_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.num_train_envs = self.all_args.num_train_envs
        self.num_eval_envs = self.all_args.num_eval_envs
        self.num_env_splits = self.all_args.num_env_splits
        # interval
        self.save_interval = self.all_args.save_interval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if not (all_args.eval or all_args.render):
            self.run_dir = run_dir
            os.makedirs(self.run_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(self.run_dir, 'log.txt'),
                                     mode='a')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)

        self.storages = storages
        self.policy = policy
        self.env_ctrls = env_ctrls
        self.info_queue = info_queue

        self.eval_storages = eval_storages
        self.eval_env_ctrls = eval_env_ctrls
        self.eval_info_queue = eval_info_queue

        if self.model_dir is not None:
            self.restore()

        self.trainer = make_trainer(self.all_args.algo,
                                    self.all_args,
                                    self.policy,
                                    archive_policies=archive_policies,
                                    archive_trajs=archive_trajs)
        # self.trainer = RKHSPO(self.all_args,
        #                       self.policy,
        #                       archive_policies=archive_policies,
        #                       archive_trajs=archive_trajs)

    def run(self):
        start = time.time()
        episodes = int(
            self.num_env_steps) // self.episode_length // self.num_train_envs

        train_ep_ret = train_ep_length = train_ep_cnt = train_win_cnt = 0

        for episode in range(episodes):
            timing = Timing()

            update_lagrangian_lambda = True
            if (episode < episodes * self.warm_up_rate):
                update_lagrangian_lambda = False

            if self.all_args.algo == 'rspo':
                # omega is the likelihood threshold multiplier, anneal it
                args = self.all_args
                omega = 1.
                progress = min(1., episode / episodes)
                if args.threshold_annealing_schedule == "linear":
                    omega = 1. - progress
                elif args.threshold_annealing_schedule == "cosine":
                    omega = np.cos(progress)
                elif args.threshold_annealing_schedule is None:
                    pass
                else:
                    raise NotImplementedError()

            for step in range(self.episode_length):
                for s_i in range(self.num_env_splits):
                    storage = self.storages[s_i]
                    assert step == storage.step

                    # Sample actions
                    with timing.add_time("envstep"):
                        for ctrl in self.env_ctrls[s_i]:
                            ctrl.obs_ready.acquire()
                            assert not ctrl.obs_ready.acquire(block=False)

                    with timing.add_time("inference"):
                        rollout_result = self.collect(s_i, step)

                    with timing.add_time("storage"):
                        storage.value_preds[step] = rollout_result.value
                        storage.actions[step] = rollout_result.action.float()
                        storage.action_log_probs[
                            step] = rollout_result.log_prob
                        if storage.policy_state is not None:
                            storage.policy_state[
                                step + 1] = rollout_result.policy_state

                        storage.step += 1

                    for ctrl in self.env_ctrls[s_i]:
                        assert not ctrl.act_ready.acquire(block=False)
                        ctrl.act_ready.release()

            with timing.add_time("bootstrap_value"):
                for s_i in range(self.num_env_splits):
                    for ctrl in self.env_ctrls[s_i]:
                        ctrl.obs_ready.acquire()

                    storage = self.storages[s_i]

                    assert storage.step == self.episode_length
                    rollout_result = self.collect(s_i, self.episode_length)
                    storage.value_preds[
                        self.episode_length] = rollout_result.value

                sample = recursive_aggregate(self.storages,
                                             lambda x: torch.cat(x, dim=1))

            if self.all_args.algo == 'rspo':
                sample = recursive_apply(sample,
                                         lambda x: x.to(self.policy.device))
                args = self.all_args
                num_refs = len(self.archive_policies)

                if num_refs > 0:
                    ex_in_rewards = self.compute_intrinsic_rewards(sample[:-1])
                    sample.rewards = sample.rewards.repeat(
                        1, 1, 1, 3 * num_refs + 1)
                    sample.rewards[:-1] = ex_in_rewards

                # compute GAE with intrinsic rewards
                if self.policy.popart_head is not None:
                    trace_target_value = self.policy.denormalize_value(
                        sample.value_preds)
                else:
                    trace_target_value = sample.value_preds

                adv, rspo_return = rspo_gae_trace(sample.rewards,
                                                  trace_target_value,
                                                  sample.masks, num_refs,
                                                  args.gamma, args.gae_lambda,
                                                  sample.bad_masks)
                assert rspo_return.shape[-1] == len(self.archive_policies)
                assert adv.shape[-1] == len(self.archive_policies) * 2 + 1

                if num_refs > 0 and episode == 0 and self.all_args.auto_alpha is not None:
                    self.likelihood_threshold = rspo_return[:-1].mean(
                        dim=(0, 1, 2)) * args.auto_alpha
                    logger.info(f"Auto tuned threshold "
                                f"{self.likelihood_threshold.cpu().numpy()}")
                    train_infos = {}
                else:
                    sample.returns = adv + trace_target_value
                    sample.advantages = adv
                    assert not torch.isnan(sample.advantages).any()
                    sample.advantages[:-1] = masked_normalization(
                        adv[:-1],
                        mask=sample.active_masks[:-1],
                        dim=tuple(range(len(adv.shape) - 1)))
                    assert not torch.isnan(sample.advantages).any()

                    if num_refs > 0:
                        interpolation_masks = self.compute_interpolation_masks(
                            omega, rspo_return[:-1])
                        sample.interpolation_masks = torch.zeros_like(adv)
                        sample.interpolation_masks[:-1] = interpolation_masks
                    else:
                        sample.interpolation_masks = torch.ones_like(
                            sample.masks)

                    with timing.add_time("train"):
                        train_infos = self.train(sample[:-1])

                    if num_refs > 0:
                        train_infos['omega'] = omega
                        train_infos[
                            'r_pred_error'] = sample.rewards[:-1, ...,
                                                             1:num_refs +
                                                             1].mean()
                        train_infos['norm_neg_l'] = sample.rewards[:-1, ...,
                                                                   1 +
                                                                   num_refs:2 *
                                                                   num_refs +
                                                                   1].mean()
                        train_infos['neg_l'] = sample.rewards[:-1, ...,
                                                              2 * num_refs +
                                                              1:].mean()
            else:
                with timing.add_time("train"):
                    if self.all_args.algo == "smerl":
                        use_intrinsic_rewards = (not train_ep_cnt == 0) and (
                            train_win_cnt / train_ep_cnt >
                            self.all_args.smerl_threshold)
                    else:
                        use_intrinsic_rewards = True
                    train_infos = self.train(
                        sample,
                        update_lagrangian_lambda=update_lagrangian_lambda,
                        use_intrinsic_rewards=use_intrinsic_rewards,
                    )

            logger.debug(timing)

            while True:
                try:
                    info = self.info_queue.get_nowait()
                    train_ep_ret += info['episode']['r']
                    train_ep_length += info['episode']['l']
                    train_ep_cnt += 1
                    if info.get('win') is not None:
                        train_win_cnt += int(info['win'])
                except queue.Empty:
                    break

            # post process
            total_num_steps = (episode +
                               1) * self.episode_length * self.num_train_envs
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            for s_i, storage in enumerate(self.storages):
                storage[0] = storage[-1]
                # storage.step must be changed via inplace operations
                assert storage.step == self.episode_length
                storage.step %= self.episode_length
                if train_ep_cnt > 0:
                    storage.avg_ep_len[:] = train_ep_length / train_ep_cnt

                for ctrl in self.env_ctrls[s_i]:
                    ctrl.obs_ready.release()

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                logger.info(
                    "Updates {}/{} episodes, total num timesteps {}/{}, FPS {}."
                    .format(episode, episodes, total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start))))

                if train_ep_cnt > 0:
                    train_env_info = dict(
                        train_episode_length=train_ep_length / train_ep_cnt)
                    if isinstance(train_ep_ret,
                                  np.ndarray) and len(train_ep_ret) > 1:
                        for k, ret in enumerate(train_ep_ret):
                            train_env_info[
                                f'train_episode_return/{k}'] = ret / train_ep_cnt
                    else:
                        train_env_info[
                            'train_episode_return'] = train_ep_ret / train_ep_cnt
                    train_infos = {**train_env_info, **train_infos}

                    if info.get('win') is not None:
                        train_win_cnt += int(info['win'])
                        train_infos[
                            'train_win_rate'] = train_win_cnt / train_ep_cnt

                    train_ep_ret = train_ep_length = train_ep_cnt = train_win_cnt = 0

                self.log_info(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 or episode == episodes - 1:
                self.eval(total_num_steps)

    @torch.no_grad()
    def collect(self, split, step, eval_=False) -> RolloutResult:
        self.trainer.prep_rollout()
        storage = self.storages[split] if not eval_ else self.eval_storages[
            split]
        request = RolloutRequest(
            storage.obs[step], storage.policy_state[step]
            if storage.policy_state is not None else None, storage.masks[step])
        request = recursive_apply(
            request, lambda x: x.flatten(end_dim=1).to(self.device))
        rollout_result = self.policy.rollout(request, deterministic=eval_)
        bs = (self.num_train_envs //
              self.num_env_splits if not eval_ else self.num_eval_envs //
              self.num_env_splits)
        return recursive_apply(
            rollout_result,
            lambda x: x.view(bs, self.num_agents, *x.shape[1:]).cpu())

    def train(self, sample, **kwargs):
        train_infos = defaultdict(lambda: 0)
        self.trainer.prep_training()
        for _ in range(self.all_args.sample_reuse):
            # `update_lagrangian_lambda` is only used for sipo-wd
            train_info = self.trainer.train(
                recursive_apply(sample, lambda x: x.to(self.device)), **kwargs)
            for k, v in train_info.items():
                train_infos[k] += v
        self.policy.inc_version()

        return {
            k: float(v / self.all_args.sample_reuse)
            for k, v in train_infos.items()
        }

    def eval(self, log_step):

        for s_i in range(self.num_env_splits):
            for ctrl in self.eval_env_ctrls[s_i]:
                while ctrl.act_ready.acquire(block=False):
                    continue
                while ctrl.obs_ready.acquire(block=False):
                    continue
                ctrl.eval_finish.clear()
                ctrl.eval_start.set()

        self.trainer.prep_rollout()
        for storage in self.eval_storages:
            storage.masks[0] = 1
            storage.bad_masks[0] = 1
            storage.active_masks[0] = 1
            if storage.policy_state is not None:
                storage.policy_state[0] = 0

        eval_ep_cnt = eval_ep_len = eval_ep_ret = eval_win_cnt = 0

        for step in range(self.all_args.eval_episode_length):
            for s_i in range(self.num_env_splits):
                storage = self.eval_storages[s_i]
                assert step == storage.step, (step, storage.step)

                for ctrl in self.eval_env_ctrls[s_i]:
                    ctrl.obs_ready.acquire()
                    assert not ctrl.obs_ready.acquire(block=False)

                rollout_result = self.collect(s_i, step, eval_=True)

                storage.value_preds[step] = rollout_result.value
                storage.actions[step] = rollout_result.action.float()
                storage.action_log_probs[step] = rollout_result.log_prob
                if storage.policy_state is not None:
                    storage.policy_state[step +
                                         1] = rollout_result.policy_state

                storage.step += 1
                storage.step %= self.all_args.eval_episode_length

                for ctrl in self.eval_env_ctrls[s_i]:
                    assert not ctrl.act_ready.acquire(block=False)
                    ctrl.act_ready.release()

                # this is due to the limited size of eval info queue
                while True:
                    try:
                        info = self.eval_info_queue.get_nowait()
                        eval_ep_ret += info['episode']['r']
                        eval_ep_len += info['episode']['l']
                        eval_ep_cnt += 1
                        if info.get('win') is not None:
                            eval_win_cnt += int(info['win'])
                    except queue.Empty:
                        break

        for s_i in range(self.num_env_splits):
            assert self.eval_storages[s_i].step == 0
            for ctrl in self.eval_env_ctrls[s_i]:
                ctrl.obs_ready.acquire()
                assert not ctrl.obs_ready.acquire(block=False)
                ctrl.eval_start.clear()
                ctrl.eval_finish.set()

        while True:
            try:
                info = self.eval_info_queue.get_nowait()
                eval_ep_ret += info['episode']['r']
                eval_ep_len += info['episode']['l']
                eval_ep_cnt += 1
                if info.get('win') is not None:
                    eval_win_cnt += int(info['win'])
            except queue.Empty:
                break

        if eval_ep_cnt > 0:
            eval_info = dict(eval_episode_length=eval_ep_len / eval_ep_cnt,
                             eval_episode_cnt=eval_ep_cnt)
            if isinstance(eval_ep_ret, np.ndarray) and len(eval_ep_ret) > 1:
                for k, ret in enumerate(eval_ep_ret):
                    eval_info[f'eval_episode_return/{k}'] = ret / eval_ep_cnt
            else:
                eval_info['eval_episode_return'] = eval_ep_ret / eval_ep_cnt

            if info.get('win') is not None:
                eval_info['eval_win_rate'] = eval_win_cnt / eval_ep_cnt

            self.log_info(eval_info, log_step)
        else:
            logger.warning("No episode finished during rollout. "
                           "Please set a larger eval_episode_length")

    def save(self):
        torch.save(
            self.policy.get_checkpoint(),
            os.path.join(str(self.run_dir), "model.pt"),
        )
        torch.save(
            recursive_aggregate(self.eval_storages,
                                lambda x: torch.cat(x, dim=1))[:-1],
            os.path.join(str(self.run_dir), "data.traj"),
        )

    def restore(self):
        checkpoint = torch.load(str(self.model_dir))
        self.policy.load_checkpoint(checkpoint)
        logger.info(f"Loaded checkpoint from {self.model_dir}.")

    def log_info(self, infos, step):
        logger.info('-' * 40)
        for k, v in infos.items():
            key = ' '.join(k.split('_')).title()
            logger.info("{}: \t{:.4f}".format(key, float(v)))
        logger.info('-' * 40)

        if not (self.all_args.eval or self.all_args.render):
            if self.all_args.use_wandb:
                wandb.log(infos,
                          step=int(step + self.iteration * self.num_env_steps))

    @torch.no_grad()
    def compute_intrinsic_rewards(self, sample):
        args = self.all_args
        episode_length, n_rollout_threads, num_agents = sample.masks.shape[:3]
        if self.policy.num_rnn_layers > 0:
            data_chunk_length = self.all_args.data_chunk_length

            num_chunks = episode_length // data_chunk_length
            assert data_chunk_length <= episode_length and episode_length % data_chunk_length == 0

            def _cast(x):
                x = x.reshape(num_chunks, x.shape[0] // num_chunks,
                              *x.shape[1:])
                x = x.transpose(1, 0)
                return x.flatten(start_dim=1, end_dim=3)

            def _unfold(x):
                x = x.view(data_chunk_length, num_chunks, n_rollout_threads,
                           num_agents, *x.shape[2:])
                x = x.transpose(0, 1)
                return x.flatten(end_dim=1)

            sample = recursive_apply(sample, _cast)
        else:

            def _unfold(x):
                return x.reshape(episode_length, n_rollout_threads, num_agents,
                                 *x.shape[2:])

            sample = recursive_apply(sample, lambda x: x.flatten(end_dim=2))
        sample = recursive_apply(sample, lambda x: x.to(self.policy.device))

        # process random slots, compute likelihood threshold
        likelihoods = []
        reward_predictions = []
        for ref_policy in self.archive_policies:
            with torch.no_grad():
                logp, _, _ = ref_policy.analyze(sample)
                r_pred = ref_policy.get_reward_prediction(
                    sample.obs.obs, sample.actions)
            likelihoods.append(logp)
            reward_predictions.append(r_pred)

        neg_l = (-torch.cat(likelihoods, -1) * args.likelihood_alpha).detach()

        assert args.use_reward_predictor
        predicted_rewards = torch.cat(reward_predictions,
                                      -1) / args.reward_prediction_multiplier
        prediction_errors = ((predicted_rewards -
                              sample.rewards).abs()).detach()
        normalized_neg_l = neg_l * 0.01

        sample = recursive_apply(sample, _unfold)
        return torch.cat([
            sample.rewards,
            recursive_apply(prediction_errors, _unfold),
            recursive_apply(normalized_neg_l, _unfold),
            recursive_apply(neg_l, _unfold)
        ], -1)

    def compute_interpolation_masks(self, omega, likelihoods):
        # cast shape
        num_refs = len(self.archive_policies)
        episode_length = self.all_args.episode_length
        n_rollout_threads, num_agents = likelihoods.shape[1:3]
        if self.policy.num_rnn_layers > 0:
            data_chunk_length = self.all_args.data_chunk_length

            num_chunks = episode_length // data_chunk_length
            assert data_chunk_length <= episode_length and episode_length % data_chunk_length == 0

            def _cast(x):
                x = x.reshape(num_chunks, x.shape[0] // num_chunks,
                              *x.shape[1:])
                x = x.transpose(1, 0)
                return x.flatten(start_dim=1, end_dim=3)

            def _unfold(x):
                x = x.view(1, num_chunks, n_rollout_threads, num_agents,
                           2 * num_refs + 1)
                x = x.repeat(data_chunk_length, 1, 1, 1, 1)
                x = x.transpose(0, 1)
                return x.flatten(end_dim=1)

            likelihoods = _cast(likelihoods)
        else:

            def _unfold(x):
                return x.reshape(episode_length, n_rollout_threads, num_agents,
                                 2 * num_refs + 1)

            likelihoods = likelihoods.view(1, -1, likelihoods.shape[-1])

        assert len(likelihoods.shape) == 3, likelihoods.shape
        args = self.all_args
        threshold = omega * self.likelihood_threshold
        num_refs = len(self.archive_policies)
        assert likelihoods.shape[-1] == num_refs, likelihoods.shape

        accepted_chunks = 0
        accepted_chunks_ref = torch.zeros(num_refs, device=self.policy.device)
        total_chunks = likelihoods.shape[1]
        for i in range(total_chunks):
            res = (likelihoods[:, i] > threshold).all(0)
            accepted_chunks += int(res.all())
            accepted_chunks_ref += res

        efficiency = accepted_chunks / total_chunks
        efficiency_ref = accepted_chunks_ref / total_chunks

        if args.no_exploration_rewards:
            exploration_rewards = False
        elif args.exploration_threshold is not None:
            exploration_rewards = efficiency < args.exploration_threshold
        else:
            exploration_rewards = True
        logger.info(
            f"accept episodes {accepted_chunks}/{total_chunks}, "
            f"efficiency {efficiency} efficiency per ref {efficiency_ref}")
        logger.info(f"use exploration rewards? {exploration_rewards}")

        alphas = [1., 0., 1.]
        use_filters = [True, False, True]

        if args.use_reward_predictor and num_refs > 0:
            alphas[1] = args.prediction_reward_alpha
            use_filters[1] = False

        if num_refs > 0:
            alphas[2] = args.exploration_reward_alpha

        if not exploration_rewards:
            alphas[2] = 0.

        shape = list(likelihoods.shape[:2])
        if num_refs == 0:
            interpolate_masks = alphas[0] * torch.ones(
                shape + [1], device=self.policy.device)
        else:
            if not any(use_filters):
                extrinsic_mask = torch.ones(shape + [1],
                                            device=self.policy.device)
                prediction_masks = torch.ones(shape + [num_refs],
                                              device=self.policy.device)
                exploration_masks = torch.ones(shape + [num_refs],
                                               device=self.policy.device)
            else:
                failed_mask = threshold > likelihoods
                if use_filters[0]:
                    extrinsic_mask = 1. - failed_mask.any(
                        dim=-1, keepdim=True).any(dim=0, keepdim=True).float()
                else:
                    extrinsic_mask = torch.ones(shape + [1],
                                                device=self.policy.device)
                if use_filters[1]:
                    prediction_masks = 1. - failed_mask.float()
                else:
                    prediction_masks = torch.ones(shape + [num_refs],
                                                  device=self.policy.device)
                if use_filters[2]:
                    exploration_masks = failed_mask.float()
                else:
                    exploration_masks = torch.ones(shape + [num_refs],
                                                   device=self.policy.device)

            interpolate_masks = torch.cat([
                alphas[0] * extrinsic_mask,
                alphas[1] * prediction_masks.any(0, keepdim=True),
                alphas[2] * exploration_masks.any(0, keepdim=True)
            ], -1)
        return _unfold(interpolate_masks)
