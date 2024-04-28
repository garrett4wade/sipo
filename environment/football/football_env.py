from gfootball.env import create_environment
from gfootball.env.wrappers import Simple115StateWrapper
import copy
import gym
import numpy as np
import os
import torch

import environment.env_base
from environment.env_utils import CentStateObservation, CentStateObservationSpace
from environment.env_wrappers import ENV_WRAPPER_KWARGS
from utils.namedarray import namedarray

# simple115v2 representation:
# - left pos (1 keeper + 10 player) --- 22
# - left vel (1 keeper + 10 player) --- 22
# - right pos (1 keeper + 10 player) --- 22
# - right vel (1 keeper + 10 player) --- 22
# - ball pos --- 3
# - ball vel --- 3
# - ball ownership (none, left, right) -- 3
# - active player --- 11
# - game mode --- 7

map_agent_registry = {
    # evn_name: (left, right, game_length, total env steps)
    # keeper is not included in controllable players
    "11_vs_11_competition": (10, 10, 3000, None),
    "11_vs_11_easy_stochastic": (10, 10, 3000, None),
    "11_vs_11_hard_stochastic": (10, 10, 3000, None),
    "11_vs_11_kaggle": (10, 10, 3000, None),
    "11_vs_11_stochastic": (10, 10, 3000, None),
    "1_vs_1_easy": (1, 1, 500, None),
    "5_vs_5": (4, 4, 3000, None),
    "academy_3_vs_1_with_keeper": (3, 1, 400, int(25e6)),
    "academy_corner": (10, 10, 400, int(50e6)),
    "academy_counterattack_easy": (10, 10, 400, int(25e6)),
    "academy_counterattack_hard": (10, 10, 400, int(50e6)),
    "academy_run_pass_and_shoot_with_keeper": (2, 1, 400, int(25e6)),
    "academy_pass_and_shoot_with_keeper": (2, 1, 400, int(25e6)),
}


@namedarray
class FootballCentStateObservation(CentStateObservation):
    ball_owned_team: np.ndarray = None
    ball_owned_player: np.ndarray = None


class FootballCentStateObservationSpace(CentStateObservationSpace):

    def sample(self):
        o = super().sample()
        return FootballCentStateObservation(
            o.obs,
            o.cent_state,
            torch.zeros(1, dtype=torch.float32, device=o.obs.device),
            torch.zeros(1, dtype=torch.float32, device=o.obs.device),
        )


class FootballEnvironment:
    """A wrapper of google football environment
    """

    def seed(self, seed):
        self.__env.seed(seed)

    def __init__(self, seed=None, share_reward=False, **kwargs):
        self.__env_name = kwargs["env_name"]
        self.__step_limit = map_agent_registry[self.__env_name][-1]
        self.__representation = "simple115v2"

        self.control_left = map_agent_registry[self.__env_name][0]
        self.control_right = 0

        # Obtain ball ownership information from raw observation.
        # Process the raw observation explicitly with wrappers
        kwargs['representation'] = "raw"

        for k in ENV_WRAPPER_KWARGS:
            if k in kwargs:
                kwargs.pop(k)

        self.__env = create_environment(
            number_of_left_players_agent_controls=self.control_left,
            number_of_right_players_agent_controls=self.control_right,
            **kwargs)
        self.seed(seed)
        self.__share_reward = share_reward

        self.__step_count = np.zeros(1, dtype=np.int32)
        self.__episode_return = np.zeros((self.num_agents, 1),
                                         dtype=np.float32)

    @property
    def n_agents(self):
        return self.num_agents

    @property
    def num_agents(self) -> int:
        return self.control_left + self.control_right

    @property
    def observation_spaces(self):
        return [
            FootballCentStateObservationSpace(
                (115, ), (map_agent_registry[self.__env_name][0] * 4 + 6, ))
            for _ in range(self.num_agents)
        ]

    @property
    def action_spaces(self):
        return [
            self.__env.action_space[0]
            if self.num_agents > 1 else self.__env.action_space
            for _ in range(self.num_agents)
        ]

    def __make_cent_state(self, obs):
        n_l_players = map_agent_registry[self.__env_name][0]
        n_r_players = map_agent_registry[self.__env_name][1]
        cent_state = np.concatenate([
            obs[..., 2:2 + n_l_players * 2], obs[..., 24:24 + n_l_players * 2],
            obs[..., 88:88 + 6]
        ], -1)
        assert (cent_state[..., 0, :] == cent_state[..., -1, :]).all()
        return cent_state

    def get_cent_state_size(self):
        return self.__make_cent_state(np.zeros((3, 115),
                                               dtype=np.float32)).shape[-1]

    def reset(self):
        obs = self.__env.reset()
        ball_owned_team = np.zeros((self.control_left, 1), dtype=np.float32)
        ball_owned_player = np.zeros((self.control_left, 1), dtype=np.float32)
        ball_owned_team[:] = obs[0]['ball_owned_team']
        ball_owned_player[:] = obs[0]['ball_owned_player']
        self.__step_count[:] = self.__episode_return[:] = 0
        obs, _ = self.__post_process_obs_and_rew(obs,
                                                 np.zeros(self.num_agents))
        return FootballCentStateObservation(
            obs,
            self.__make_cent_state(obs),
            ball_owned_team,
            ball_owned_player,
        )

    def __post_process_obs_and_rew(self, obs, reward):
        assert self.__representation == "simple115v2"
        if self.num_agents == 1:
            obs = obs[np.newaxis, :]
            reward = [reward]
        # if self.__representation == "extracted":
        #     obs = np.swapaxes(obs, 1, 3)
        if self.__representation in ("simple115", "simple115v2"):
            obs = Simple115StateWrapper.convert_observation(
                obs, (self.__representation == 'simple115v2'))
            obs[obs == -1] = 0
        if self.__share_reward:
            left_reward = np.mean(reward[:self.control_left])
            if self.control_right > 0:
                right_reward = np.mean(reward[self.control_left:])
            else:
                right_reward = 0
            reward = np.array([left_reward] * self.control_left +
                              [right_reward] * self.control_right)
        return obs, reward

    def step(self, actions):
        assert len(actions) == self.num_agents, len(actions)
        obs, reward, done, info = self.__env.step([int(a) for a in actions])
        ball_owned_team = np.zeros((self.control_left, 1), dtype=np.float32)
        ball_owned_player = np.zeros((self.control_left, 1), dtype=np.float32)
        ball_owned_team[:] = obs[0]['ball_owned_team']
        ball_owned_player[:] = obs[0]['ball_owned_player']
        obs, reward = self.__post_process_obs_and_rew(obs, reward)
        self.__step_count += 1
        self.__episode_return += reward[:, np.newaxis]
        info['win'] = (info['score_reward'] > 0)
        info['episode'] = dict(r=self.__episode_return.mean().item(),
                               l=self.__step_count.item())
        info['bad_transition'] = (
            done and self.__step_count.item() >= self.__step_limit)
        return (
            FootballCentStateObservation(
                obs,
                self.__make_cent_state(obs),
                ball_owned_team,
                ball_owned_player,
            ),
            np.array(reward[:, None], dtype=np.float32),
            np.array([[done] for _ in range(self.num_agents)], dtype=np.uint8),
            [copy.deepcopy(info) for _ in range(self.num_agents)],
        )

    def render(self, mode='human'):
        return self.__env.render(mode=mode)

    def close(self):
        self.__env.close()


environment.env_base.register("football", FootballEnvironment)
