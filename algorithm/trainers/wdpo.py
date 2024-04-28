from collections import defaultdict
import numpy as np
import math
import torch
import torch.nn as nn

from algorithm.modules import gae_trace, masked_normalization
from algorithm.trainer import feed_forward_generator, recurrent_generator, register
from algorithm.trainers.mappo import MAPPO
from utils.namedarray import recursive_apply


class WDPO(MAPPO):

    def __init__(self, args, policy, archive_policies, archive_trajs):
        super().__init__(args, policy)

        self.discriminator_optimizers = []
        for i, ref_p in enumerate(archive_policies):
            opt = torch.optim.RMSprop(ref_p.discriminator.parameters(),
                                      lr=args.discriminator_lr,
                                      eps=args.opti_eps,
                                      weight_decay=args.weight_decay)
            self.discriminator_optimizers.append(opt)

        self.lagrangian_lambda = nn.Parameter(torch.zeros(
            len(archive_policies), device=self.policy.device),
                                              requires_grad=False)
        self.lagrangian_lr = args.lagrangian_lr
        self.threshold_eps = args.threshold_eps
        self.ll_max = args.ll_max  # ll means 'L'agrangian 'L'ambda

        self.archive_policies = archive_policies
        self.archive_trajs = archive_trajs
        self.i_r_scaling = args.intrinsic_reward_scaling

    def train(self,
              storage,
              update_actor=True,
              update_lagrangian_lambda=True,
              radio=1.0,
              **kwargs):
        train_info = defaultdict(lambda: 0)
        wdpo_info = {}

        if len(self.archive_policies) > 0:
            # compute wasserstein distance and intrinsic rewards, optimize discriminators
            i_r = torch.zeros_like(storage.rewards[:-1]).repeat(
                1, 1, 1, len(self.archive_policies))
            i_return = torch.zeros(len(self.archive_policies)).to(i_r)
            for ref_i, (ref_policy, traj) in enumerate(
                    zip(self.archive_policies, self.archive_trajs)):
                bs0 = storage.masks.shape[0]
                bs1 = storage.masks.shape[1]
                if (traj.masks.shape[0] >= bs0):
                    indices0 = np.sort(
                        np.random.choice(traj.masks.shape[0],
                                         bs0,
                                         replace=False))
                else:
                    indices0 = np.sort(
                        np.random.choice(traj.masks.shape[0],
                                         bs0,
                                         replace=True))

                if (traj.masks.shape[1] >= bs1):
                    indices1 = np.sort(
                        np.random.choice(traj.masks.shape[1],
                                         bs1,
                                         replace=False))
                else:
                    indices1 = np.sort(
                        np.random.choice(traj.masks.shape[1],
                                         bs1,
                                         replace=True))

                if (not self.args.use_cross_entropy):
                    w_dist = ref_policy.w_dist(
                        recursive_apply(traj[indices0][:, indices1][:-1],
                                        lambda x: x.to(self.policy.device)),
                        storage[:-1])

                    self.discriminator_optimizers[ref_i].zero_grad()
                    (-w_dist).mean().backward()
                    self.discriminator_optimizers[ref_i].step()
                    for p in ref_policy.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    w_dist.detach_()
                    # temporal difference
                    _w_dist = torch.cat(
                        [torch.zeros_like(w_dist[0:1]), w_dist[:-1]], dim=0)
                    i_r[..., ref_i:ref_i +
                        1] = w_dist - _w_dist * storage.masks[:-1]
                else:
                    new_log_probs, _, _ = ref_policy.analyze(storage[:-1])
                    new_log_probs.detach_()
                    i_r[..., ref_i:ref_i + 1] = -new_log_probs

            # update lagrangian lambda
            if (self.args.use_gda):
                i_return = i_r.sum(
                    0, keepdim=True).mean(dim=tuple(range(len(i_r.shape) - 1)))
                assert i_return.shape == self.lagrangian_lambda.shape

                # storage.rewards = torch.cat([storage.rewards, i_r], -1)
                storage.rewards[:-1] += (self.lagrangian_lambda * i_r *
                                         self.i_r_scaling).sum(-1,
                                                               keepdim=True)

                if (update_lagrangian_lambda):
                    self.lagrangian_lambda = self.lagrangian_lambda + self.lagrangian_lr * (
                        (self.args.threshold_eps * i_r.shape[0] - i_return))
                    self.lagrangian_lambda.clamp_(0, self.args.ll_max * radio)

                for ref_i in range(len(self.archive_policies)):
                    wdpo_info[
                        f"lagrangian_lambda/{ref_i}"] = self.lagrangian_lambda[
                            ref_i].float()
                    wdpo_info[f"intrinsic_return/{ref_i}"] = i_return[
                        ref_i].float() / i_r.shape[0]

                wdpo_info['avg_intrinsic_reward'] = (self.lagrangian_lambda *
                                                     i_r *
                                                     self.i_r_scaling).sum(
                                                         -1,
                                                         keepdim=True).mean()
                wdpo_info['threshold'] = self.args.threshold_eps

            if (self.args.use_filter):
                i_return = i_r.sum(0, keepdim=True)
                for ref_i in range(len(self.archive_policies)):
                    wdpo_info[f"intrinsic_return/{ref_i}"] = (i_return.mean(
                        dim=tuple(range(len(i_r.shape) -
                                        1)))[ref_i]).float() / i_r.shape[0]
                intrisic_bound = (i_return > self.args.threshold_eps *
                                  i_r.shape[0]).all(dim=-1, keepdim=True)

                wdpo_info['avg_intrinsic_reward'] = ((~intrisic_bound) * (
                    (i_return <= self.args.threshold_eps * i_r.shape[0]) *
                    i_r).sum(-1, keepdim=True)).mean().float()
                wdpo_info['avg_extrinsic_reward'] = (
                    intrisic_bound * storage.rewards[:-1]).mean().float()
                storage.rewards[:-1] = (~intrisic_bound) * (
                    (i_return <= self.args.threshold_eps * i_r.shape[0]) * i_r
                ).sum(-1, keepdim=True) + intrisic_bound * storage.rewards[:-1]

        # compute GAE with intrinsic rewards
        if self.policy.popart_head is not None:
            trace_target_value = self.policy.denormalize_value(
                storage.value_preds)
        else:
            trace_target_value = storage.value_preds
        adv = gae_trace(storage.rewards, trace_target_value, storage.masks,
                        self.args.gamma, self.args.gae_lambda,
                        storage.bad_masks)
        storage.returns = adv + trace_target_value
        storage.advantages = adv
        storage.advantages[:-1] = masked_normalization(
            adv[:-1], mask=storage.active_masks[:-1])

        for _ in range(self.ppo_epoch):
            if self.policy.num_rnn_layers > 0:
                data_generator = recurrent_generator(storage[:-1],
                                                     self.num_mini_batch,
                                                     self.data_chunk_length)
            else:
                data_generator = feed_forward_generator(
                    storage[:-1], self.num_mini_batch)

            for sample in data_generator:
                (value_loss, critic_grad_norm, policy_loss, dist_entropy,
                 actor_grad_norm,
                 imp_weights) = self.ppo_update(sample,
                                                update_actor=update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        train_info = {**train_info, **wdpo_info}

        return train_info


register("sipo-wd", WDPO)