#!/usr/bin/env python
from pathlib import Path
import copy
import itertools
import logging
import multiprocessing as mp
import os
import torch

from algorithm.trainer import SampleBatch
from configs.config import make_config
from environment.env_wrappers import EnvironmentControl
from environment.eval_env_worker import shared_eval_worker
from runner.shared_runner import SharedRunner
from utils.namedarray import recursive_apply, array_like
from environment.eval_env_worker import TrajReplayBuffer

logger = logging.getLogger('WinTraj')

N_WIN_TRAJECTORIES_TO_COLLECT = 32
WIN_TRAJECTORY_MAX_LENGTH = 200

def get_critic_dim(algo, iteration):
    if algo == 'sipo-rbf':
        return iteration + 1
    if algo == 'rspo':
        return 2 * iteration + 1
    return 1

def collect_win_traj(all_args, obs_space, act_dim, policy, iteration=None):
    all_args = copy.deepcopy(all_args)
    all_args.eval = True
    all_args.render = False
    all_args.eval_episode_length = 100 if all_args.env_name == "football" else 400
    config = make_config(all_args.config, all_args.algo)
    environment_config = config['environment']

    max_episode_length = WIN_TRAJECTORY_MAX_LENGTH
    n_trajs = N_WIN_TRAJECTORIES_TO_COLLECT
    num_agents = all_args.num_agents

    traj_buffer = TrajReplayBuffer(
        n_trajs,
        max_episode_length,
        all_args.num_agents,
        SampleBatch(
            # NOTE: sampled available actions should be 1
            obs=array_like(obs_space.sample()),
            value_preds=None,
            actions=torch.zeros(act_dim),
            action_log_probs=torch.zeros(1),
            rewards=torch.zeros(1),
            masks=torch.zeros(1),
            active_masks=torch.zeros(1),
            bad_masks=torch.zeros(1),
        ),
    )
    traj_queue = mp.Queue(maxsize=n_trajs)

    eval_storages = []
    critic_dim = 1 if iteration is None else get_critic_dim(all_args.algo, iteration)
    for _ in range(all_args.num_env_splits):
        eval_storage = SampleBatch(
            # NOTE: sampled available actions should be 1
            obs=obs_space.sample(),
            value_preds=torch.zeros(critic_dim),
            actions=torch.zeros(act_dim),
            action_log_probs=torch.zeros(1),
            rewards=torch.zeros(1),
            masks=torch.ones(1),
            active_masks=torch.ones(1),
            bad_masks=torch.ones(1),
            warmup_masks=torch.ones(1),
        )
        if policy.num_rnn_layers > 0:
            eval_storage.policy_state = policy.policy_state_space.sample()
        eval_storage = recursive_apply(
            eval_storage,
            lambda x: x.repeat(
                all_args.eval_episode_length + 1, all_args.num_eval_envs //
                all_args.num_env_splits, num_agents, *(
                    (1, ) * len(x.shape))).share_memory_(),
        )
        eval_storage.step = torch.tensor(0, dtype=torch.long).share_memory_()
        eval_storage.avg_ep_len = torch.tensor(
            [0], dtype=torch.long).share_memory_()
        eval_storages.append(eval_storage)

    eval_env_ctrls = [[
        EnvironmentControl(mp.Semaphore(0), mp.Semaphore(0), mp.Event(),
                           mp.Event(), mp.Event())
        for _ in range(all_args.n_eval_rollout_threads //
                       all_args.num_env_splits)
    ] for _ in range(all_args.num_env_splits)]
    eval_info_queue = mp.Queue(all_args.n_eval_rollout_threads)

    # start worker
    envs_per_worker = all_args.num_eval_envs // all_args.n_eval_rollout_threads
    eval_workers = [[
        mp.Process(
            target=shared_eval_worker,
            args=(
                i,
                [environment_config for _ in range(envs_per_worker)],
                eval_env_ctrls[j][i],
                eval_storages[j],
                eval_info_queue,
            ),
            kwargs=dict(traj_queue=traj_queue),
        ) for i in range(all_args.n_eval_rollout_threads //
                         all_args.num_env_splits)
    ] for j in range(all_args.num_env_splits)]
    for ew in itertools.chain.from_iterable(eval_workers):
        ew.start()

    runner = SharedRunner(
        0,
        all_args,
        policy,
        [],
        [],
        [],
        [],
        None,
        eval_storages,
        eval_env_ctrls,
        eval_info_queue,
        policy.device,
    )
    runner.eval(0)

    while not traj_queue.empty():
        traj = traj_queue.get_nowait()
        traj_buffer.put(traj)

    logger.info(f"{traj_buffer.size()} win trajectories are collected!")
    traj_buffer.shuntdown()

    # post process
    for ctrl in itertools.chain(*eval_env_ctrls):
        ctrl.exit_.set()
    for worker in itertools.chain(*eval_workers):
        worker.join()

    return traj_buffer.storage[:, :traj_buffer.size()]


def collect_win_traj_smerl(all_args, obs_space, act_dim, policy):
    all_args = copy.deepcopy(all_args)
    all_args.eval = True
    all_args.render = False
    all_args.eval_episode_length = WIN_TRAJECTORY_MAX_LENGTH
    config = make_config(all_args.config, all_args.algo)
    environment_config = config['environment']
    latent_dim = environment_config['args']['base']['latent_dim']

    max_episode_length = WIN_TRAJECTORY_MAX_LENGTH
    n_trajs = N_WIN_TRAJECTORIES_TO_COLLECT
    num_agents = all_args.num_agents

    all_win_trajs = []
    for latent_idx in range(latent_dim):
        environment_config = make_config(all_args.config, all_args.algo)['environment']
        environment_config['args']['base']['fix_latent_idx'] = latent_idx
        traj_buffer = TrajReplayBuffer(
            n_trajs,
            max_episode_length,
            all_args.num_agents,
            SampleBatch(
                # NOTE: sampled available actions should be 1
                obs=array_like(obs_space.sample()),
                value_preds=None,
                actions=torch.zeros(act_dim),
                action_log_probs=torch.zeros(1),
                rewards=torch.zeros(1),
                masks=torch.zeros(1),
                active_masks=torch.zeros(1),
                bad_masks=torch.zeros(1),
            ),
        )
        traj_queue = mp.Queue(maxsize=n_trajs)

        eval_storages = []
        for _ in range(all_args.num_env_splits):
            eval_storage = SampleBatch(
                # NOTE: sampled available actions should be 1
                obs=obs_space.sample(),
                value_preds=torch.zeros(1),
                actions=torch.zeros(act_dim),
                action_log_probs=torch.zeros(1),
                rewards=torch.zeros(1),
                masks=torch.ones(1),
                active_masks=torch.ones(1),
                bad_masks=torch.ones(1),
                warmup_masks=torch.ones(1),
            )
            if policy.num_rnn_layers > 0:
                eval_storage.policy_state = policy.policy_state_space.sample()
            eval_storage = recursive_apply(
                eval_storage,
                lambda x: x.repeat(
                    all_args.eval_episode_length + 1, all_args.num_eval_envs //
                    all_args.num_env_splits, num_agents, *(
                        (1, ) * len(x.shape))).share_memory_(),
            )
            eval_storage.step = torch.tensor(0, dtype=torch.long).share_memory_()
            eval_storage.avg_ep_len = torch.tensor(
                [0], dtype=torch.long).share_memory_()
            eval_storages.append(eval_storage)

        eval_env_ctrls = [[
            EnvironmentControl(mp.Semaphore(0), mp.Semaphore(0), mp.Event(),
                            mp.Event(), mp.Event())
            for _ in range(all_args.n_eval_rollout_threads //
                        all_args.num_env_splits)
        ] for _ in range(all_args.num_env_splits)]
        eval_info_queue = mp.Queue(all_args.n_eval_rollout_threads)

        # start worker
        envs_per_worker = all_args.num_eval_envs // all_args.n_eval_rollout_threads
        eval_workers = [[
            mp.Process(
                target=shared_eval_worker,
                args=(
                    i,
                    [environment_config for _ in range(envs_per_worker)],
                    eval_env_ctrls[j][i],
                    eval_storages[j],
                    eval_info_queue,
                ),
                kwargs=dict(traj_queue=traj_queue),
            ) for i in range(all_args.n_eval_rollout_threads //
                            all_args.num_env_splits)
        ] for j in range(all_args.num_env_splits)]
        for ew in itertools.chain.from_iterable(eval_workers):
            ew.start()

        runner = SharedRunner(
            0,
            all_args,
            policy,
            [],
            [],
            [],
            [],
            None,
            eval_storages,
            eval_env_ctrls,
            eval_info_queue,
            policy.device,
        )
        runner.eval(0)

        while not traj_queue.empty():
            traj = traj_queue.get_nowait()
            traj_buffer.put(traj)

        logger.info(f"{traj_buffer.size()} win trajectories are collected!")
        traj_buffer.shuntdown()

        # post process
        for ctrl in itertools.chain(*eval_env_ctrls):
            ctrl.exit_.set()
        for worker in itertools.chain(*eval_workers):
            worker.join()

        all_win_trajs.append(traj_buffer.storage[:, :traj_buffer.size()])
    return all_win_trajs