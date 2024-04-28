import torch
import torch.nn as nn
import gym


class FrameStackWassersteinDiscriminator(nn.Module):

    def __init__(self, obs_dim, act_dim, act_space, hidden_dim, **kwargs):
        super().__init__()
        # NOTE: action is currently not used for discrimination
        self.obs_dim = obs_dim
        self.act_space = act_space
        self.act_dim = act_dim
        self._input_embedding = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),  # obs + rewards
            nn.ReLU())
        self._linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # obs + rewards
            nn.ReLU())
        self._out = nn.Linear(hidden_dim, 1)
        self._hidden_dim = hidden_dim

    def forward(self, obs: torch.Tensor, actions: torch.Tensor,
                rewards: torch.Tensor, masks: torch.Tensor):
        # if (isinstance(self.act_space, gym.spaces.Discrete)):
        #     actions = nn.functional.one_hot(actions.long(), self.act_dim)
        #     actions = actions.squeeze(dim=-2)
        # x = envirs = torch.cat([obs, rewards], dim=-1)
        # x = torch.cat([envirs, actions], dim=-1)
        x = self._linear(self._input_embedding(obs))
        return self._out(x)


def WassersteinDiscriminator(type_, *args, **kwargs):
    if type_ == 'frame_stack':
        d = FrameStackWassersteinDiscriminator(*args, **kwargs)
    else:
        raise NotImplementedError(
            f"Discriminator type {type_} not implemented.")
    for p in d.parameters():
        p.data.clamp_(-0.01, 0.01)
    return d
