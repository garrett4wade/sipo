from collections import deque
from typing import Optional
import copy
import dataclasses
import logging
import gym
import multiprocessing as mp
import numpy as np
import random
import time
import torch
import os
import queue

from algorithm.trainer import SampleBatch
from utils.namedarray import recursive_aggregate, recursive_apply, array_like
import environment.env_base as env_base
from environment.env_wrappers import TorchTensorWrapper, EnvironmentControl, _check_shm


class TrajReplayBuffer:
    """A shared-memory trajectory buffer."""

    def __init__(
        self,
        buffer_size,
        max_episode_length,
        num_agents,
        example_sample_batch,
    ):

        self._maxsize = buffer_size
        self._max_episode_length = max_episode_length
        self._num_agents = num_agents

        self._ptr = torch.zeros((1, ), dtype=torch.int64)
        self._cur_size = torch.zeros((1, ), dtype=torch.int64)

        self._storage = recursive_apply(
            example_sample_batch,
            lambda x: x.repeat(max_episode_length, self._maxsize, num_agents,
                               *((1, ) * len(x.shape))),
        )

        self._init_values = recursive_apply(self._storage,
                                            lambda x: x.flatten()[0].clone())

    @property
    def storage(self):
        return self._storage

    def size(self):
        return int(self._cur_size)

    def full(self):
        return self._cur_size == self._maxsize

    def put(self, traj):
        traj_len = traj.masks.shape[0]
        if traj_len > self._max_episode_length:
            return
        idx = int(self._ptr)
        self._ptr[:] = (self._ptr + 1) % self._maxsize
        self._cur_size[:] = min(self._maxsize, self._cur_size + 1)
        self._storage[:, idx] = self._init_values
        self._storage[:traj_len, idx] = traj

    def shuntdown(self):
        pass


def shared_eval_worker(
    rank,
    environment_configs,
    env_ctrl: EnvironmentControl,
    storage: SampleBatch,
    info_queue: mp.Queue,
    traj_queue: mp.Queue,
    warmup_fraction: Optional[float] = 0.0,
):

    recursive_apply(storage, _check_shm)

    offset = rank * len(environment_configs)
    envs = []
    for i, cfg in enumerate(environment_configs):
        if cfg.get('args'):
            if cfg['args'].get('base'):
                cfg['args']['base']['seed'] = random.randint(0, int(1e6))
            else:
                cfg['args']['base'] = dict(seed=random.randint(0, int(1e6)))
        else:
            cfg['args'] = {'base': dict(seed=random.randint(0, int(1e6)))}
        env = TorchTensorWrapper(env_base.make(cfg, split='eval'))
        envs.append(env)

    traj_storages = [[] for _ in range(len(environment_configs))]

    while not env_ctrl.exit_.is_set():

        if not env_ctrl.eval_start.is_set():
            time.sleep(1)
            continue

        for i, env in enumerate(envs):
            obs = env.reset()
            storage.obs[0, offset + i] = obs
        env_ctrl.obs_ready.release()

        while not env_ctrl.eval_finish.is_set():

            if env_ctrl.act_ready.acquire(timeout=0.1):
                step = storage.step
                avg_ep_len = storage.avg_ep_len

                for i, env in enumerate(envs):
                    if step - 1 < 0:
                        continue
                    act = storage.actions[step - 1, offset + i]
                    obs, reward, done, info = env.step(act)

                    done_env = done.all(0, keepdim=True).float()
                    mask = 1 - done_env

                    active_mask = 1 - done
                    active_mask = active_mask * (1 - done_env) + done_env

                    bad_mask = torch.tensor(
                        [[0.0] if info_.get('bad_transition') else [1.0]
                         for info_ in info],
                        dtype=torch.float32)

                    warmup_mask = torch.tensor(
                        [[0.0] if info_['episode']['l'] <= warmup_fraction *
                         avg_ep_len else [1.0] for info_ in info],
                        dtype=torch.float32)

                    traj_step = SampleBatch(
                        obs=storage.obs[step - 1, offset + i],
                        value_preds=None,
                        action_log_probs=None,
                        actions=act,
                        rewards=reward,
                        masks=storage.masks[step - 1, offset + i],
                        active_masks=storage.active_masks[step - 1,
                                                          offset + i],
                        bad_masks=storage.bad_masks[step - 1, offset + i],
                    )
                    if len(traj_storages[i]) == 0:
                        traj_step.masks = torch.ones_like(traj_step.masks)
                        traj_step.active_masks = torch.ones_like(
                            traj_step.active_masks)
                    traj_storages[i].append(traj_step)

                    if done.all():
                        if info[0].get('win'):
                            final_traj_step = SampleBatch(
                                obs=obs,
                                action_log_probs=None,
                                value_preds=None,
                                actions=torch.zeros_like(act),
                                rewards=torch.zeros_like(reward),
                                masks=torch.zeros_like(active_mask),
                                active_masks=torch.zeros_like(active_mask),
                                bad_masks=torch.zeros_like(bad_mask),
                            )
                            traj = recursive_aggregate(traj_storages[i] +
                                                       [final_traj_step],
                                                       torch.stack)  # [T, D]
                            try:
                                traj_queue.put_nowait(traj)
                            except queue.Full:
                                pass

                        traj_storages[i] = []

                        obs = env.reset()
                        try:
                            info_queue.put_nowait(info[0])
                        except queue.Full:
                            pass

                    storage.obs[step, offset + i] = obs
                    storage.rewards[step - 1, offset + i] = reward
                    storage.masks[step, offset + i] = mask
                    storage.active_masks[step, offset + i] = active_mask
                    storage.bad_masks[step, offset + i] = bad_mask
                    storage.warmup_masks[step - 1, offset + i] = warmup_mask

                env_ctrl.obs_ready.release()

    for env in envs:
        env.close()
