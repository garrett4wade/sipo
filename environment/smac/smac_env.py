"""Wrapper around the SMAC environments provided by https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/envs/starcraft2/StarCraft2_Env.py.
"""
from typing import List, Tuple, Dict, Union, Optional
import collections
import copy
import dataclasses
import gym
import logging
import numpy as np
import os
import time
import torch
import warnings

from environment.smac.smac_env_ import StarCraft2Env
from utils.namedarray import namedarray, recursive_apply
import utils.numpy_utils
import environment.env_base as env_base
import environment.env_utils as env_utils

logger = logging.getLogger("env-smac")


def get_smac_shapes(map_name,
                    use_state_agent=True,
                    agent_specific_obs=False,
                    agent_specific_state=False):
    env = StarCraft2Env(map_name, use_state_agent=use_state_agent)
    return env.get_obs_size(agent_specific_obs), env.get_state_size(
        agent_specific_state), env.get_total_actions(), env.n_agents


@namedarray
class SMACAgentSpecificObs(env_base.Observation):
    obs_allies: np.ndarray
    obs_enemies: np.ndarray
    obs_move: np.ndarray
    obs_self: np.ndarray
    obs_mask: np.ndarray


@namedarray
class SMACAgentSpecificState(env_base.Observation):
    state_allies: np.ndarray
    state_enemies: np.ndarray
    state_self: np.ndarray
    state_mask: np.ndarray
    state_move: Optional[np.ndarray] = None


@namedarray
class SMACObservation(env_base.Observation):
    obs: Union[np.ndarray, SMACAgentSpecificObs]
    state: Union[np.ndarray, SMACAgentSpecificState]
    available_action: np.ndarray


@namedarray
class SMACCentStateObservation(SMACObservation):
    cent_state: Optional[np.ndarray] = None


class SMACEnvironment(env_base.Environment):

    def __init__(self,
                 map_name,
                 save_replay=False,
                 agent_specific_obs=False,
                 agent_specific_state=False,
                 **kwargs):
        self.map_name = map_name
        self.__save_replay = save_replay
        self.__agent_specific_obs = agent_specific_obs
        self.__agent_specific_state = agent_specific_state

        # Initializing the SMAC environment can continuously fail if the game client tries
        # to connect to an already-in-use port. The resolution is to start multiple trials.
        for i in range(10):
            try:
                self.__env = StarCraft2Env(map_name=map_name, **kwargs)
                self.__act_space = gym.spaces.Discrete(self.__env.n_actions)
                obs, state, available_action, _, _ = self.__env.reset()

                actions = []
                for i in range(self.__env.n_agents):
                    act = self.__act_space.sample()
                    while not available_action[i, act]:
                        act = self.__act_space.sample()
                    actions.append(act)

                self.__env.step(np.array(actions))
                break
            except Exception as e:
                print(
                    f"Failed to start SC2 Environment due to {e}, retrying {i}"
                )
        else:
            raise RuntimeError("Failed to start SC2.")

        self.__obs_shapes = self.__env.get_obs_size(
            agent_specific=self.__agent_specific_obs)
        self.__state_shapes = self.__env.get_state_size(
            agent_specific=self.__agent_specific_state)

        if self.__agent_specific_obs:
            self.__obs_split_shapes = copy.deepcopy(self.__obs_shapes)
            self.__obs_split_shapes.pop('obs_mask')
        if self.__agent_specific_state:
            self.__state_split_shapes = copy.deepcopy(self.__state_shapes)
            self.__state_split_shapes.pop('state_mask')

        cent_state = self.__env.get_cent_state()
        print(cent_state.shape)
        self.__obs_space = env_utils.ObservationSpaceFromExample(
            recursive_apply(
                SMACCentStateObservation(obs[0], state[0],
                                         available_action[0],
                                         cent_state),
                lambda x: torch.from_numpy(x).float()))

    @property
    def agent_count(self) -> int:
        # consider smac as a single-agent environment if the action/observation spaces are shared
        return self.__env.n_agents

    @property
    def num_agents(self):
        return self.agent_count

    @property
    def n_agents(self):
        return self.agent_count

    @property
    def observation_spaces(self) -> List[dict]:
        return [self.__obs_space for _ in range(self.agent_count)]

    @property
    def action_spaces(self) -> List[dict]:
        return [self.__act_space for _ in range(self.agent_count)]

    def reset(self):
        local_obs, state, available_action, obs_mask, state_mask = self.__env.reset(
        )
        cent_state = self.__env.get_cent_state()
        cent_state = np.stack([cent_state for _ in range(self.__env.n_agents)],
                              -2)

        if self.__agent_specific_obs:
            local_obs = SMACAgentSpecificObs(
                **utils.numpy_utils.split_to_shapes(local_obs,
                                                    self.__obs_split_shapes,
                                                    -1),
                obs_mask=obs_mask)
        if self.__agent_specific_state:
            state = SMACAgentSpecificState(**utils.numpy_utils.split_to_shapes(
                state, self.__state_split_shapes, -1),
                                           state_mask=state_mask)

        return SMACCentStateObservation(local_obs, state, available_action,
                                        cent_state)

    def step(self, actions):
        assert actions.shape == (self.agent_count, 1), actions.shape

        (local_obs, state, rewards, dones, infos, available_action, obs_mask,
         state_mask) = self.__env.step(actions)
        cent_state = self.__env.get_cent_state()
        cent_state = np.stack([cent_state for _ in range(self.__env.n_agents)],
                              -2)

        if self.__agent_specific_obs:
            local_obs = SMACAgentSpecificObs(
                **utils.numpy_utils.split_to_shapes(local_obs,
                                                    self.__obs_split_shapes,
                                                    -1),
                obs_mask=obs_mask)
        if self.__agent_specific_state:
            state = SMACAgentSpecificState(**utils.numpy_utils.split_to_shapes(
                state, self.__state_split_shapes, -1),
                                           state_mask=state_mask)

        if self.__save_replay and np.all(dones):
            self.__env.save_replay()

        assert rewards.shape == (len(available_action), 1), rewards.shape
        assert dones.shape == (len(available_action), 1), dones.shape
        for info in infos:
            info['episode'] = dict(r=float(info['episode_return']),
                                   l=int(info['episode_length']))
        obs = SMACCentStateObservation(local_obs, state, available_action,
                                       cent_state)
        return obs, rewards, dones, infos

    def render(self, mode=None) -> None:
        raise NotImplementedError(
            'Rendering the SMAC environment is by default disabled. '
            'Please run evaluation with "save_replay=True" instead.')

    def seed(self, seed):
        self.__env.seed(seed)
        return seed

    def close(self):
        self.__env.close()


env_base.register("smac", SMACEnvironment)