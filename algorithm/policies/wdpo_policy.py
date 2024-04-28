import itertools
import gym
import torch

from algorithm import modules
from algorithm.policies.actor_critic_policy import ActorCriticPolicy
from algorithm.policy import register


class WDPOPolicy(ActorCriticPolicy):

    def __init__(self,
                 observation_space,
                 action_space,
                 discriminator_type='frame_stack',
                 discriminator_hidden_dim: int = 64,
                 hidden_dim: int = 64,
                 device=torch.device("cuda:0"),
                 *args,
                 **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs)
        if isinstance(action_space, gym.spaces.Discrete):
            act_dim = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            act_dim = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            act_dim = action_space.nvec

        x = observation_space.sample()
        cent_state_dim = x.cent_state.shape[-1]

        self.discriminator = modules.WassersteinDiscriminator(
            discriminator_type, cent_state_dim, act_dim, action_space,
            discriminator_hidden_dim, **kwargs).to(device)

    def parameters(self):
        return itertools.chain(self.actor.parameters(),
                               self.critic.parameters(),
                               self.discriminator.parameters())

    def load_checkpoint(self, checkpoint):
        self._version = checkpoint.get("steps", 0)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint['critic'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

    def get_checkpoint(self):
        return {
            "steps": self._version,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "discriminator": self.discriminator.state_dict()
        }

    def w_dist(self, self_traj, other_traj):
        r1 = self.discriminator(self_traj.obs.cent_state, self_traj.actions,
                                self_traj.rewards, self_traj.masks)
        r2 = self.discriminator(other_traj.obs.cent_state, other_traj.actions,
                                other_traj.rewards, other_traj.masks)
        return r1 - r2


register("wdpo", WDPOPolicy)