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

    assert not (all_args.eval or all_args.render)
    if all_args.n_rollout_threads is None:
        all_args.n_rollout_threads = all_args.num_train_envs
    if all_args.n_eval_rollout_threads is None:
        all_args.n_eval_rollout_threads = all_args.num_eval_envs
    assert all_args.num_train_envs % all_args.n_rollout_threads == 0
    assert all_args.num_eval_envs % all_args.n_eval_rollout_threads == 0
    assert all_args.n_rollout_threads % all_args.num_env_splits == 0
    assert all_args.n_eval_rollout_threads % all_args.num_env_splits == 0

    if all_args.algo == 'smerl':
        assert all_args.n_iterations == 1
        assert len(all_args.archive_policy_dirs) == 0
        assert len(all_args.archive_traj_dirs) == 0

    policy_config = config['policy']
    environment_config = config['environment']
    all_args.env_name = environment_config['type']

    logger.info("all config: {}".format(all_args))
    if not all_args.seed_specify:
        all_args.seed = np.random.randint(1000, 10000)
    if 'SLURM_PROCID' in os.environ:
        all_args.seed += int(os.environ['SLURM_PROCID'])
        all_args.wandb_name = f"seed{all_args.seed}"
    assert len(all_args.archive_policy_dirs) <= all_args.n_iterations
    assert len(all_args.archive_traj_dirs) <= all_args.n_iterations

    logger.info("seed is: {}".format(all_args.seed))
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

    run_dir = (Path(f"results/{all_args.algo}") / all_args.config /
               all_args.experiment_name / str(all_args.seed))
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    exst_run_nums = [
        int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir()
        if str(folder.name).startswith('run')
    ]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.wandb_project
                         if all_args.wandb_project else all_args.config,
                         group=all_args.wandb_group
                         if all_args.wandb_group else all_args.experiment_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=all_args.wandb_name if all_args.wandb_name else
                         str(all_args.experiment_name) + "_seed" +
                         str(all_args.seed),
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)

    setproctitle.setproctitle(
        str(all_args.config) + "-" + str(all_args.experiment_name) + "@" +
        str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

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

    archive_policies = []
    archive_trajs = []
    for i, (policy_dir, traj_dir) in enumerate(
            zip(all_args.archive_policy_dirs, all_args.archive_traj_dirs)):
        policy_config_ = copy.deepcopy(policy_config)
        policy_config_['args']['critic_dim'] = get_critic_dim(all_args.algo, i)
        policy = algorithm.policy.make(policy_config_, obs_space, act_space)
        policy.load_checkpoint(torch.load(str(policy_dir)))
        archive_policies.append(policy)

        traj: SampleBatch = torch.load(str(traj_dir))
        archive_trajs.append(traj)

    for iteration in range(len(all_args.archive_policy_dirs),
                           all_args.n_iterations):
        logger.info(
            "#" * 20 +
            f" Iteration {iteration + 1}/{all_args.n_iterations} started. " +
            "#" * 20)
        if policy_config.get('args'):
            policy_config['args']['critic_dim'] = critic_dim = get_critic_dim(
                all_args.algo, iteration)
        else:
            policy_config['args'] = dict(
                critic_dim=get_critic_dim(all_args.algo, iteration))
            critic_dim = iteration + 1
        policy = algorithm.policy.make(policy_config, obs_space, act_space)
        if all_args.inherit_policy and iteration > 0:
            policy.actor.load_state_dict(
                archive_policies[-1].get_checkpoint()["actor"])

        # initialize rollout storage
        storages = []
        for _ in range(all_args.num_env_splits):
            # initialze storage
            storage = SampleBatch(
                # NOTE: sampled available actions should be 1
                obs=obs_space.sample(),
                value_preds=torch.zeros(critic_dim),
                actions=torch.zeros(act_dim),
                action_log_probs=torch.zeros(1),
                rewards=torch.zeros(1),
                masks=torch.ones(1),
                active_masks=torch.ones(1),
                bad_masks=torch.ones(1),
                warmup_masks=torch.ones(1),
            )

            if policy.num_rnn_layers > 0:
                storage.policy_state = policy.policy_state_space.sample()

            storage = recursive_apply(
                storage,
                lambda x: x.repeat(
                    all_args.episode_length + 1, all_args.num_train_envs //
                    all_args.num_env_splits, num_agents, *(
                        (1, ) * len(x.shape))).share_memory_(),
            )
            storage.step = torch.tensor(0, dtype=torch.long).share_memory_()
            storage.avg_ep_len = torch.tensor(
                [0], dtype=torch.long).share_memory_()

            storages.append(storage)

        eval_storages = []
        for _ in range(all_args.num_env_splits):
            eval_storage = SampleBatch(
                # NOTE: sampled available actions should be 1
                obs=obs_space.sample(),
                value_preds=torch.zeros(critic_dim),
                actions=torch.zeros(act_dim),
                action_log_probs=torch.zeros(1),
                rewards=torch.zeros(1),
                masks=torch.ones(1),
                active_masks=torch.ones(1),
                bad_masks=torch.ones(1),
                warmup_masks=torch.ones(1),
            )
            if policy.num_rnn_layers > 0:
                eval_storage.policy_state = policy.policy_state_space.sample()
            eval_storage = recursive_apply(
                eval_storage,
                lambda x: x.repeat(
                    all_args.eval_episode_length + 1, all_args.num_eval_envs //
                    all_args.num_env_splits, num_agents, *(
                        (1, ) * len(x.shape))).share_memory_(),
            )
            eval_storage.step = torch.tensor(0,
                                             dtype=torch.long).share_memory_()
            eval_storage.avg_ep_len = torch.tensor(
                [0], dtype=torch.long).share_memory_()
            eval_storages.append(eval_storage)

        # initialize communication utilities
        env_ctrls = [[
            EnvironmentControl(mp.Semaphore(0), mp.Semaphore(0), mp.Event())
            for _ in range(all_args.n_rollout_threads //
                           all_args.num_env_splits)
        ] for _ in range(all_args.num_env_splits)]
        eval_env_ctrls = [[
            EnvironmentControl(mp.Semaphore(0), mp.Semaphore(0), mp.Event(),
                               mp.Event(), mp.Event())
            for _ in range(all_args.n_eval_rollout_threads //
                           all_args.num_env_splits)
        ] for _ in range(all_args.num_env_splits)]
        info_queue = mp.Queue(1000)
        eval_info_queue = mp.Queue(all_args.n_eval_rollout_threads)

        # start worker
        envs_per_worker = all_args.num_train_envs // all_args.n_rollout_threads
        env_workers = [[
            mp.Process(
                target=shared_env_worker,
                args=(
                    i,
                    [environment_config for _ in range(envs_per_worker)],
                    env_ctrls[j][i],
                    storages[j],
                    info_queue,
                ),
                kwargs=dict(warmup_fraction=all_args.warmup_fraction),
            ) for i in range(all_args.n_rollout_threads //
                             all_args.num_env_splits)
        ] for j in range(all_args.num_env_splits)]
        for worker in itertools.chain.from_iterable(env_workers):
            worker.start()

        envs_per_worker = all_args.num_eval_envs // all_args.n_eval_rollout_threads
        eval_workers = [[
            mp.Process(
                target=shared_eval_worker,
                args=(
                    i,
                    [environment_config for _ in range(envs_per_worker)],
                    eval_env_ctrls[j][i],
                    eval_storages[j],
                    eval_info_queue,
                ),
                kwargs=dict(warmup_fraction=all_args.warmup_fraction),
            ) for i in range(all_args.n_eval_rollout_threads //
                             all_args.num_env_splits)
        ] for j in range(all_args.num_env_splits)]
        for ew in itertools.chain.from_iterable(eval_workers):
            ew.start()

        logger.info("Setup finishes! Preparing to run...")
        # run experiments
        runner = SharedRunner(
            iteration,
            all_args,
            policy,
            archive_policies,
            archive_trajs,
            storages,
            env_ctrls,
            info_queue,
            eval_storages,
            eval_env_ctrls,
            eval_info_queue,
            device,
            run_dir=run_dir /
            f"iter{iteration}" if all_args.algo != 'smerl' else run_dir)

        runner.run()

        # post process
        for ctrl in itertools.chain(*env_ctrls, *eval_env_ctrls):
            ctrl.exit_.set()
        for worker in itertools.chain(*env_workers, *eval_workers):
            worker.join()

        archive_policies.append(policy)
        traj = torch.load(os.path.join(str(runner.run_dir), "data.traj"))
        archive_trajs.append(traj)

        if all_args.algo == 'smerl':
            all_win_trajs = collect_win_traj_smerl(all_args, obs_space,
                                                   act_dim, policy)
            for wt_i, win_trajs in enumerate(all_win_trajs):
                fn = os.path.join(runner.run_dir, f"win_traj{wt_i}.pt")
                torch.save(win_trajs, fn)
                logger.info(
                    f'Win trajectories {wt_i} saved at {fn}. '
                    f"Number of win trajectories: {win_trajs.masks.shape[1]}.")
        else:
            win_trajs = collect_win_traj(all_args, obs_space, act_dim, policy, iteration=iteration)
            fn = os.path.join(runner.run_dir, "win_traj.pt")
            torch.save(win_trajs, fn)
            logger.info(
                f'Win trajectories saved at {fn}. '
                f"Number of win trajectories: {win_trajs.masks.shape[1]}.")

    if not (all_args.eval or all_args.render):
        if all_args.use_wandb:
            run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
