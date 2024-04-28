import torch
import torch.nn as nn

from algorithm.policies.actor_critic_policy import ActorCriticPolicy
from algorithm.policy import register


class Discriminator(nn.Module):

    def __init__(self, cent_state_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cent_state_dim, hidden_dim),
                                 nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, x):
        return self.net(x)


class SMERLPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        latent_dim,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, **kwargs)

        cent_state_dim = observation_space.sample().cent_state.shape[-1]
        self.discriminator = Discriminator(cent_state_dim, latent_dim).to(self.device)
        self.latent_dim = latent_dim

    def get_latent_probs(self, obs, cent_state):
        onehot_latent = obs[..., -self.latent_dim:]
        latent = onehot_latent.argmax(-1)
        logits = self.discriminator(cent_state)
        return torch.distributions.Categorical(logits=logits).log_prob(latent)


register("smerl", SMERLPolicy)