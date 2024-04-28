from collections import defaultdict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.modules import gae_trace, masked_normalization
from algorithm.trainer import feed_forward_generator, recurrent_generator, register
from algorithm.trainers.mappo import MAPPO
from utils.namedarray import recursive_apply


def rbf_kernel(x, y, gamma, mask=None):
    """Compute E_x[exp(-|x-y|^2 / (2 * gamma * dim))] for each y

    Args:
        x (torch.Tensor): shape [N, D]
        y (torch.Tensor): shape [M, D]
        mask (torch.Tensor, optional): mask on x. 
            shape [N, 1]. Defaults to None.

    Returns:
        torch.Tensor: shape [M]
    """
    x_dim = x.shape[-1]

    dist = x.pow(2).sum(
        -1, keepdim=True) + y.pow(2).sum(-1) - 2 * x @ y.T  # [N, M]

    factor = 2 * gamma * x_dim
    if mask is not None:
        return ((-dist / factor).exp() * mask).sum(0) / (mask.sum() + 1e-5)
    else:
        return (-dist / factor).exp().mean(0)


@torch.no_grad()
def gaussian_dist(
    self_traj,
    other_traj,
    gamma,
    use_action=False,
    act_dim=None,
):
    self_obs = self_traj.obs.cent_state
    other_obs = other_traj.obs.cent_state

    if use_action:
        if self_traj.actions.shape[-1] == 1:
            # discrete action
            self_obs = torch.cat([
                self_obs,
                F.one_hot(self_traj.actions.squeeze(-1).long(),
                          act_dim).float()
            ],
                                 dim=-1)
            other_obs = torch.cat([
                other_obs,
                F.one_hot(other_traj.actions.squeeze(-1).long(),
                          act_dim).float()
            ],
                                  dim=-1)
        else:
            # continuous action
            self_obs = torch.cat([self_obs, self_traj.actions], dim=-1)
            other_obs = torch.cat([other_obs, other_traj.actions], dim=-1)

    shape = other_obs.shape

    self_obs = self_obs.flatten(end_dim=-2)  # [N, D]
    other_obs = other_obs.flatten(end_dim=-2)  # [M, D]

    # normalize to [-1, 1]
    min_ = torch.min(self_obs.min(dim=0).values, other_obs.min(dim=0).values)
    max_ = torch.max(self_obs.max(dim=0).values, other_obs.max(dim=0).values)
    self_obs = 2 * (self_obs - min_) / (max_ - min_ + 1e-5) - 1
    other_obs = 2 * (other_obs - min_) / (max_ - min_ + 1e-5) - 1

    warmup_mask = self_traj.warmup_masks.flatten(end_dim=-2)  # [N, 1]

    # part1 = rbf_kernel(self_obs, self_obs, gamma, mask=warmup_mask).mean()
    part2 = rbf_kernel(self_obs, other_obs, gamma, mask=warmup_mask)
    # part3 = rbf_kernel(other_obs, other_obs, gamma)
    # assert len(part2.shape) == 1 and len(part3.shape) == 1

    # diff = self_obs.unsqueeze(-2) - other_obs  # [N, M, D]
    # dist = torch.matmul(diff.unsqueeze(-2),
    #                     diff.unsqueeze(-1)).squeeze()  #[N, M]

    return -(part2 + 1e-5).log().view(*shape[:-1], 1)


class RKHSPO(MAPPO):

    def __init__(self, args, policy, archive_trajs, **kwargs):
        super().__init__(args, policy)

        self.lagrangian_lambda = nn.Parameter(torch.zeros(
            len(archive_trajs), device=self.policy.device),
                                              requires_grad=False)
        self.lagrangian_lr = args.lagrangian_lr
        self.threshold_eps = args.threshold_eps
        self.ll_max = args.ll_max  # ll means 'L'agrangian 'L'ambda

        self.archive_trajs = archive_trajs
        self.i_r_scaling = args.intrinsic_reward_scaling

        self.rkhs_action = args.rkhs_action

    def train(self, storage, update_actor=True, **kwargs):
        train_info = defaultdict(lambda: 0)
        rkhs_info = {}

        ll = self.lagrangian_lambda.clone()

        if len(self.archive_trajs) > 0:
            # compute intrinsic rewards
            i_r = torch.zeros(storage.rewards.shape,
                              dtype=torch.float32).repeat(
                                  1, 1, 1, len(self.archive_trajs))
            for ref_i, traj in enumerate(self.archive_trajs):
                i_r[:-1, ..., ref_i:ref_i + 1] = gaussian_dist(
                    traj,
                    recursive_apply(storage[:-1], lambda x: x.cpu()),
                    use_action=self.rkhs_action,
                    act_dim=self.policy.actor.action_dim,
                    gamma=self.args.rbf_gamma)

            i_r *= self.i_r_scaling
            i_r = i_r.to(self.policy.device)
            assert not torch.isnan(i_r).any()

            # update lagrangian lambda
            # TODO: homeostasis
            i_return = i_r.sum(
                0, keepdim=True).mean(dim=tuple(range(len(i_r.shape) - 1)))
            assert i_return.shape == self.lagrangian_lambda.shape
            self.lagrangian_lambda = self.lagrangian_lambda + self.lagrangian_lr * (
                self.args.threshold_eps * self.i_r_scaling * i_r.shape[0] -
                i_return)
            self.lagrangian_lambda.clamp_(0, self.args.ll_max)

            storage.rewards = torch.cat([storage.rewards, i_r], -1)

            for ref_i in range(len(self.archive_trajs)):
                rkhs_info[
                    f"lagrangian_lambda/{ref_i}"] = self.lagrangian_lambda[
                        ref_i].float()
                rkhs_info[f"intrinsic_return/{ref_i}"] = i_return[ref_i]
            rkhs_info['avg_intrinsic_reward'] = i_r.sum(-1,
                                                        keepdim=True).mean()
            rkhs_info[
                'threshold'] = self.args.threshold_eps * self.i_r_scaling * i_r.shape[
                    0]

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
        assert not torch.isnan(storage.advantages).any()
        storage.advantages[:-1] = masked_normalization(
            adv[:-1],
            mask=storage.active_masks[:-1],
            dim=tuple(range(len(adv.shape) - 1)))
        assert not torch.isnan(storage.advantages).any()

        # sum extrinsic & intrinsic rewards
        storage.advantages = storage.advantages[..., 0:1] + (
            ll * storage.advantages[..., 1:]).sum(-1, keepdim=True)

        # storage.advantages[:-1] = masked_normalization(
        #     storage.advantages[:-1], mask=storage.active_masks[:-1])
        assert not torch.isnan(storage.advantages).any()

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
        train_info = {**train_info, **rkhs_info}

        return train_info

register("sipo-rbf", RKHSPO)