from collections import defaultdict
import math
import torch
import torch.nn as nn

from algorithm.modules import gae_trace, masked_normalization
from algorithm.trainer import SampleBatch, feed_forward_generator, recurrent_generator, register


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm()**2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


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
        return ((-dist / factor).exp() * mask).sum() / (mask.sum() + 1e-5)
    else:
        return (-dist / factor).exp().mean()


class DIPG:

    def __init__(self, args, policy, archive_policies, archive_trajs,
                 **kwargs):
        self.args = args

        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.archive_trajs = archive_trajs

        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=args.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(),
            lr=args.critic_lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay,
        )

    def cal_value_loss(self, values, value_preds_batch, return_batch,
                       active_masks_batch):
        if self.policy.popart_head is not None:
            self.policy.update_popart(return_batch)
            return_batch = self.policy.normalize_value(return_batch)

        value_pred_clipped = value_preds_batch + (
            values - value_preds_batch).clamp(-self.clip_param,
                                              self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss *
                          active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample: SampleBatch, update_iter, update_actor=True):
        # Reshape to do in a single forward pass for all steps
        action_log_probs, values, dist_entropy = self.policy.analyze(sample)
        # actor update
        imp_weights = torch.exp(action_log_probs - sample.action_log_probs)

        surr1 = imp_weights * sample.advantages
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * sample.advantages
        assert surr1.shape[-1] == surr2.shape[-1] == 1

        if self._use_policy_active_masks:
            policy_loss = (-torch.min(surr1, surr2) * sample.active_masks
                           ).sum() / sample.active_masks.sum()
            dist_entropy = (dist_entropy * sample.active_masks
                            ).sum() / sample.active_masks.sum()
        else:
            policy_loss = -torch.min(surr1, surr2).mean()
            dist_entropy = dist_entropy.mean()

        value_loss = self.cal_value_loss(values, sample.value_preds,
                                         sample.returns, sample.active_masks)

        mmd_loss = 0
        if update_iter == 0 and len(self.archive_trajs) > 0:
            kernels = []
            for ref_i, traj in enumerate(self.archive_trajs):
                kernel_value = rbf_kernel(
                    traj.obs.cent_state.flatten(end_dim=-2),
                    sample.obs.cent_state.flatten(end_dim=-2).cpu(),
                    gamma=self.args.rbf_gamma,
                    mask=traj.warmup_masks.flatten(end_dim=-2),
                )
                kernels.append(kernel_value)

            idx = int(torch.tensor(kernels).argmax())

            x = sample.obs.cent_state.flatten(end_dim=-2).cpu()
            y = self.archive_trajs[idx].obs.cent_state.flatten(end_dim=-2)
            dist = x.pow(2).sum(
                -1, keepdim=True) + y.pow(2).sum(-1) - 2 * x @ y.T  # [N, M]

            k_term = (-1.0 * dist.mean(1)).exp().to(self.policy.device)
            assert action_log_probs.flatten().shape == k_term.shape, (
                action_log_probs.shape, k_term.shape)
            mmd_loss = (action_log_probs.flatten() * (2 - 2 * k_term)).mean()

        self.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef +
             self.args.mmd_alpha * mmd_loss).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, mmd_loss

    def train(self, storage, update_actor=True, **kwargs):
        train_info = defaultdict(lambda: 0)

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
            adv[:-1], mask=storage.active_masks[:-1])
        assert not torch.isnan(storage.advantages).any()
        assert storage.advantages.shape[-1] == 1, storage.advantages.shape

        for i in range(self.ppo_epoch):
            if self.policy.num_rnn_layers > 0:
                data_generator = recurrent_generator(storage[:-1],
                                                     self.num_mini_batch,
                                                     self.data_chunk_length)
            else:
                data_generator = feed_forward_generator(
                    storage[:-1], self.num_mini_batch)

            for sample in data_generator:
                (value_loss, critic_grad_norm, policy_loss, dist_entropy,
                 actor_grad_norm, imp_weights,
                 mmd_loss) = self.ppo_update(sample,
                                             i,
                                             update_actor=update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                if len(self.archive_trajs) > 0:
                    train_info['mmd_loss'] += float(mmd_loss)

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            if k != 'mmd_loss':
                train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

register("dipg", DIPG)