from collections import defaultdict
import math
import torch
import torch.nn as nn

from algorithm.trainer import SampleBatch, feed_forward_generator, recurrent_generator, register
from algorithm.trainers.mappo import MAPPO
from algorithm.modules import gae_trace, masked_normalization


class SMERL(MAPPO):

    def __init__(self, args, policy, **kwargs):
        super().__init__(args, policy)
        self.discriminator_optimizer = torch.optim.Adam(
            self.policy.discriminator.parameters(), lr=args.lr)

    def ppo_update(self, sample: SampleBatch, update_actor=True):
        ppo_stats = super().ppo_update(sample, update_actor)
        latent_logp = self.policy.get_latent_probs(sample.obs.obs,
                                                   sample.obs.cent_state)

        self.discriminator_optimizer.zero_grad()
        d_loss = (-latent_logp).mean()
        d_loss.backward()
        self.discriminator_optimizer.step()
        return (*ppo_stats, d_loss)

    def train(self, storage, use_intrinsic_rewards, update_actor=True, **kwargs):
        train_info = defaultdict(lambda: 0)
        rkhs_info = {}

        if use_intrinsic_rewards:
            # compute intrinsic rewards
            with torch.no_grad():
                i_r = self.policy.get_latent_probs(
                    storage.obs.obs,
                    storage.obs.cent_state,
                ) + math.log(self.policy.latent_dim)
            i_r *= self.args.intrinsic_reward_scaling
            i_r = i_r.to(self.policy.device)
            assert not torch.isnan(i_r).any()

            storage.rewards = i_r.unsqueeze(-1) + storage.rewards

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
        assert storage.advantages.shape[-1] == 1, storage.advantages.shape
        storage.advantages[:-1] = masked_normalization(
            adv[:-1],
            mask=storage.active_masks[:-1],
            dim=tuple(range(len(adv.shape) - 1)))
        assert not torch.isnan(storage.advantages).any()

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
                 actor_grad_norm, imp_weights,
                 d_loss) = self.ppo_update(sample, update_actor=update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['discriminator_loss'] += d_loss.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info


register("smerl", SMERL)