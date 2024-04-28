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
import environment.env_utils as env_utils

logger = logging.getLogger('shared_runner')
logger.setLevel(logging.INFO)


class TorchTensorWrapper(gym.Wrapper):

    def __init__(self, env, device='cpu'):
        super().__init__(env)
        self._device = device

    def _to_tensor(self, x):
        return recursive_apply(x,
                               lambda y: torch.from_numpy(y).to(self._device))

    def step(self, action):
        obs, r, d, info = self.env.step(action.cpu().numpy())
        return (*list(
            map(lambda x: recursive_apply(x, self._to_tensor), [obs, r, d])),
                info)

    def reset(self):
        return recursive_apply(self.env.reset(), self._to_tensor)


# TODO: rename as env_worker.py
@dataclasses.dataclass
class EnvironmentControl:
    act_ready: mp.Semaphore
    obs_ready: mp.Semaphore
    exit_: mp.Event
    eval_start: Optional[mp.Event] = None
    eval_finish: Optional[mp.Event] = None


def _check_shm(x):
    assert isinstance(x, torch.Tensor) and x.is_shared


def shared_env_worker(rank,
                      environment_configs,
                      env_ctrl: EnvironmentControl,
                      storage: SampleBatch,
                      info_queue: mp.Queue,
                      warmup_fraction: Optional[float] = 0.0):

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
        env = TorchTensorWrapper(env_base.make(cfg, split='train'))
        envs.append(env)

    for i, env in enumerate(envs):
        obs = env.reset()
        storage.obs[0, offset + i] = obs
    env_ctrl.obs_ready.release()

    while not env_ctrl.exit_.is_set():

        if env_ctrl.act_ready.acquire(timeout=0.1):
            step = storage.step
            avg_ep_len = storage.avg_ep_len
            for i, env in enumerate(envs):
                act = storage.actions[step - 1, offset + i]
                obs, reward, done, info = env.step(act)
                if done.all():
                    obs = env.reset()
                    try:
                        info_queue.put_nowait(info[0])
                    except queue.Full:
                        pass

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

                storage.obs[step, offset + i] = obs
                storage.rewards[step - 1, offset + i] = reward
                storage.masks[step, offset + i] = mask
                storage.active_masks[step, offset + i] = active_mask
                storage.bad_masks[step, offset + i] = bad_mask
                storage.warmup_masks[step - 1, offset + i] = warmup_mask

            env_ctrl.obs_ready.release()

    for env in envs:
        env.close()


def shared_eval_worker(rank,
                       environment_configs,
                       env_ctrl: EnvironmentControl,
                       storage: SampleBatch,
                       info_queue: mp.Queue,
                       warmup_fraction: Optional[float] = 0.0,
                       render=False,
                       render_mode='rgb_array',
                       render_idle_time=0.0,
                       save_video=False,
                       video_file='output.mp4',
                       video_fps=24,
                       target_video_size=(200, 200)):

    recursive_apply(storage, _check_shm)
    if render:
        assert len(environment_configs) == 1
        if save_video:
            assert render_mode == 'rgb_array'

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
        env = TorchTensorWrapper(
            env_base.make(cfg, split=('eval' if not render else "render")))
        envs.append(env)

    frames = []

    while not env_ctrl.exit_.is_set():

        if not env_ctrl.eval_start.is_set():
            time.sleep(1)
            continue

        for i, env in enumerate(envs):
            obs = env.reset()
            storage.obs[0, offset + i] = obs
            if render:
                frames.append(env.render(mode=render_mode).astype(np.uint8))
                time.sleep(render_idle_time)
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
                    if render:
                        frames.append(
                            env.render(mode=render_mode).astype(np.uint8))
                        time.sleep(render_idle_time)

                    if done.all():
                        obs = env.reset()
                        try:
                            info_queue.put_nowait(info[0])
                        except queue.Full:
                            pass

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

                    storage.obs[step, offset + i] = obs
                    storage.rewards[step - 1, offset + i] = reward
                    storage.masks[step, offset + i] = mask
                    storage.active_masks[step, offset + i] = active_mask
                    storage.bad_masks[step, offset + i] = bad_mask
                    storage.warmup_masks[step - 1, offset + i] = warmup_mask

                env_ctrl.obs_ready.release()

    if save_video:
        video_format = video_file.split('.')[-1]
        import cv2

        if video_format == 'avi' or video_format == 'mp4':

            h, w = target_video_size
            for i, frame in enumerate(frames):
                if frame.shape[:-1] != (h, w):
                    frames[i] = cv2.resize(frame, (w, h),
                                           interpolation=cv2.INTER_AREA)

            fourcc = cv2.VideoWriter_fourcc(
                *("XVID" if video_format == 'avi' else "mp4v"))
            video = cv2.VideoWriter(video_file,
                                    fourcc,
                                    fps=video_fps,
                                    frameSize=(w, h))

            [video.write(frame) for frame in frames]

            if os.environ.get('DISPLAY') is not None:
                cv2.destroyAllWindows()
            video.release()
        elif video_format == 'gif':
            from PIL import Image
            frames = [
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                for frame in frames
            ]
            frames[0].save(video_file,
                           save_all=True,
                           append_images=frames[1:],
                           optimize=False,
                           duration=1000 / video_fps,
                           loop=0)
        else:
            raise NotImplementedError(
                f"Video format {video_format} not implemented.")

        logger.info(f"Video saved at {video_file}.")

    for env in envs:
        env.close()


ENV_WRAPPER_KWARGS = [
    'frame_stack', 'stack_key', 'num_stack', 'latent_dim', 'latent_type',
    'latent_mean', 'latent_std', 'latent_concat_keys', 'fix_latent_idx',
]


class FrameStackedObservationSpace:

    def __init__(self, sample_obs, num_stack, stack_key='discr_obs'):
        self.sample_obs = array_like(sample_obs)
        self.sample_obs[stack_key] = torch.cat(
            [self.sample_obs[stack_key] for _ in range(num_stack)], -1)
        self._shapes = {k: v.shape for k, v in self.sample_obs.items()}

    @property
    def shapes(self):
        return self._shapes

    def sample(self):
        return array_like(self.sample_obs, constructor=torch.randn_like)


class FrameStackWrapper(gym.core.Wrapper):
    """Stack the last 'num_stack' frames of the corresponding key.
    
    Basically a repreduction of
    https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py.
    """

    def __init__(self,
                 env,
                 num_stack,
                 stack_key='discr_obs',
                 wrap_action=False,
                 **kwargs):
        super().__init__(env)
        self.num_stack = num_stack
        self.stack_key = stack_key

        self.frames = deque(maxlen=num_stack)

    @property
    def observation_spaces(self):
        return [
            FrameStackedObservationSpace(space.sample(),
                                         self.num_stack,
                                         stack_key=self.stack_key)
            for space in self.env.observation_spaces
        ]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        [
            self.frames.append(obs[self.stack_key])
            for _ in range(self.num_stack)
        ]
        obs[self.stack_key] = np.concatenate(list(self.frames), -1)
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        self.frames.append(obs[self.stack_key])
        obs[self.stack_key] = np.concatenate(list(self.frames), -1)
        return obs, rewards, dones, infos


class LatentVariableWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 latent_dim,
                 latent_type='categorical',
                 latent_mean=None,
                 latent_std=None,
                 latent_concat_keys=['obs'],
                 fix_latent_idx=None,
                 **kwargs):
        super().__init__(env)
        assert latent_type == 'categorical'
        self.latent_dim = latent_dim
        self.latent_concat_keys = latent_concat_keys
        self.fix_latent_idx = fix_latent_idx

    @property
    def observation_spaces(self):
        sample = self.env.observation_spaces[0].sample()
        latent = torch.zeros(*sample.obs.shape[:-1],
                             self.latent_dim).to(sample.obs)
        latent[..., 0] = 1
        for k in self.latent_concat_keys:
            sample[k] = torch.cat([sample[k], latent], -1)
        obs_space = env_utils.ObservationSpaceFromExample(sample)
        return [obs_space for _ in range(self.env.num_agents)]

    def reset(self):
        obs = self.env.reset()
        self.episode_latent = np.zeros((*obs.obs.shape[:-1], self.latent_dim),
                                       dtype=np.float32)
        if self.fix_latent_idx is None:
            self.episode_latent[...,
                                random.randint(0, self.latent_dim - 1)] = 1
        else:
            self.episode_latent[..., self.fix_latent_idx] = 1
        for k in self.latent_concat_keys:
            obs[k] = np.concatenate([obs[k], self.episode_latent], -1)
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        for k in self.latent_concat_keys:
            obs[k] = np.concatenate([obs[k], self.episode_latent], -1)
        return obs, rewards, dones, infos