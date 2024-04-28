#!/usr/bin/env python
from pathlib import Path
import copy
import gym
import itertools
import logging
import multiprocessing as mp
import numpy as np
import os
import setproctitle
import socket
import sys
import torch
import wandb
import yaml

from algorithm.trainer import SampleBatch
from apps.collect_win_traj import collect_win_traj, collect_win_traj_smerl
from configs.config import get_base_config, make_config
from environment.env_wrappers import shared_env_worker, shared_eval_worker, EnvironmentControl, TorchTensorWrapper
from runner.shared_runner import SharedRunner
from utils.namedarray import recursive_apply, recursive_aggregate
import algorithm.policy
import environment.env_base as env_base

logging.basicConfig(
    format=
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


def get_critic_dim(algo, iteration):
    if algo == 'sipo-rbf':
        return iteration + 1
    if algo == 'rspo':
        return 2 * iteration + 1
    return 1


def main(args):
    parser = get_base_config()
    all_args = parser.parse_known_args(args)[0]
    config = make_config(all_args.config, all_args.algo)
    for k, v in config.get("base", {}).items():
        if f"--{k}" not in args:
            setattr(all_args, k, v)
        else:
            logger.warning(f"CLI argument {k} conflicts with yaml config. "
                           f"The latter will be overwritten "
                           f"by CLI arguments {k}={getattr(all_args, k)}.")

    assert all_args.model_dir is not None

    algos = ['sipo-rbf', 'sipo-wd', 'rspo', 'dipg', 'smerl']
    assert sum([int(algo in all_args.model_dir) for algo in algos]) == 1
    assert not (all_args.eval or all_args.render)

    if all_args.n_eval_rollout_threads is None:
        all_args.n_eval_rollout_threads = all_args.num_eval_envs
    assert all_args.num_eval_envs % all_args.n_eval_rollout_threads == 0
    assert all_args.n_eval_rollout_threads % all_args.num_env_splits == 0

    for algo in algos:
        if algo in all_args.model_dir:
            all_args.algo = algo
            break

    policy_config = config['policy']
    environment_config = config['environment']
    all_args.env_name = environment_config['type']

    # cuda
    torch.set_num_threads(os.cpu_count())
    if all_args.cuda and torch.cuda.is_available():
        logger.info("choose to use gpu...")
        device = torch.device("cuda:0")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        logger.info("choose to use cpu...")
        device = torch.device("cpu")

    run_dir = os.path.abspath(os.path.dirname(all_args.model_dir))

    # env
    example_env = TorchTensorWrapper(
        env_base.make(environment_config, split='train'), device)
    act_space = example_env.action_spaces[0]
    obs_space = example_env.observation_spaces[0]
    all_args.num_agents = num_agents = example_env.num_agents
    del example_env

    if isinstance(act_space, gym.spaces.Discrete):
        act_dim = 1
    elif isinstance(act_space, gym.spaces.Box):
        act_dim = act_space.shape[0]
    elif isinstance(act_space, gym.spaces.MultiDiscrete):
        act_dim = len(act_space.nvec)
    else:
        raise NotImplementedError()

    iteration = None
    if "iter" in all_args.model_dir:
        iteration = int(all_args.model_dir.split("iter")[-1].split('/')[0].split('.')[0])

    if policy_config.get('args'):
        policy_config['args']['critic_dim'] = critic_dim = get_critic_dim(
            all_args.algo, iteration)
    else:
        policy_config['args'] = dict(
            critic_dim=get_critic_dim(all_args.algo, iteration))
        critic_dim = iteration + 1
    policy = algorithm.policy.make(policy_config, obs_space, act_space)

    if all_args.algo == 'smerl':
        all_win_trajs = collect_win_traj_smerl(all_args, obs_space,
                                                act_dim, policy)
        for wt_i, win_trajs in enumerate(all_win_trajs):
            fn = os.path.join(run_dir, f"win_traj{wt_i}.pt")
            torch.save(win_trajs, fn)
            logger.info(
                f'Win trajectories {wt_i} saved at {fn}. '
                f"Number of win trajectories: {win_trajs.masks.shape[1]}.")
    else:
        win_trajs = collect_win_traj(all_args, obs_space, act_dim, policy, iteration=iteration)
        fn = os.path.join(run_dir, "win_traj.pt")
        torch.save(win_trajs, fn)
        logger.info(
            f'Win trajectories saved at {fn}. '
            f"Number of win trajectories: {win_trajs.masks.shape[1]}.")


if __name__ == "__main__":
    main(sys.argv[1:])
