from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import PatchCollection
import argparse
import collections
import cv2
import logging
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import scipy.signal
import sys
import time
import torch
import torch.distributed
import torch.nn as nn

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument("--paradigm",
                    type=str,
                    choices=['pbt', 'iter', 'both', 'render'])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--pbt_constraint_type",
                    type=str,
                    choices=['min', 'pairwise'],
                    default='pairwise')
parser.add_argument("--n_landmarks", type=int, default=4)
parser.add_argument("--landmark_size", type=float, default=0.10)
parser.add_argument("--agent_size", type=float, default=0.05)
parser.add_argument("--eps", type=float, default=1e-5)
parser.add_argument("--min_landmark_dist", type=float, default=0.16)
parser.add_argument("--max_landmark_dist", type=float, default=1.0)
parser.add_argument("--render_name", type=str, default="env_demo.png")
args = parser.parse_args()

n_landmarks = args.n_landmarks
landmark_size = args.landmark_size
agent_size = args.agent_size
eps = args.eps
min_landmark_dist = args.min_landmark_dist
max_landmark_dist = args.max_landmark_dist

n_save_trajs = 10
pbt_traj_dir = "./pbt_trajs"
iter_traj_dir = "./iter_trajs"

# n_landmarks = 3
# landmark_size = 0.10
# agent_size = 0.05
# eps = 1e-5
# min_landmark_dist = 0.4
# max_landmark_dist = 1.2

# landmarks will be initialized in a circular ring
# with inner radius=min_landmark_dist and outer radius=max_landmark_dist
constraint_agent_dist = 2 * (agent_size + landmark_size + eps) + 0.1
world_size = max_landmark_dist + landmark_size + eps
constraint_landmark_dist = constraint_agent_dist + 2 * (agent_size +
                                                        landmark_size + eps)
assert min_landmark_dist > agent_size + landmark_size + eps

delta_t = 0.1

device = torch.device("cpu")
if args.paradigm == 'iter':
    device = torch.device("cuda")
distributed_backend = 'gloo' if device == torch.device("cpu") else 'nccl'

lagrangian_lr = 0.5
ll_max = 10.0

max_ep_len = 1000


def l2_dist(x1, x2):
    """Return pairwise L2 distance between two matrices.

    Args:
        x1 (torch.Tensor): shape [M, D]
        x2 (torch.Tensor): shape [N, D]

    Returns:
        pairwise distance: shape [M, N]
    """
    return torch.sum(x1 ** 2, dim=1, keepdim=True) + \
               torch.sum(x2 ** 2, dim=1) - \
               2 * torch.matmul(x1, x2.t())


def log_info(infos, step, rank=None):
    if rank is not None:
        logger.info('-' * 12 + f" Rank {rank} " + '-' * 12)
    else:
        logger.info('-' * 40)
    for k, v in infos.items():
        key = ' '.join(k.split('_')).title()
        logger.info("{}: \t{:.4f}".format(key, float(v)))
    logger.info('-' * 40)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    return torch.from_numpy(
        scipy.signal.lfilter([1], [1, float(-discount)],
                             x.cpu().numpy()[::-1],
                             axis=0)[::-1].copy()).to(x)


def generate_landmark_pos():
    landmark_pos = []

    cnt = 0
    while len(landmark_pos) < n_landmarks:
        if cnt >= n_landmarks * 1000:
            raise RuntimeError("Too many trials for landmark initialization!")
        cnt += 1
        pos = (torch.rand(2) - 0.5) * 2 * (
            max_landmark_dist - min_landmark_dist) + min_landmark_dist
        if min_landmark_dist is not None and pos.norm() < min_landmark_dist:
            continue
        if max_landmark_dist is not None and pos.norm() > max_landmark_dist:
            continue
        if any([(pos - p).norm() < constraint_landmark_dist
                for p in landmark_pos]):
            continue
        landmark_pos.append(pos)
    return torch.stack(landmark_pos)


class NaviEnv:

    def __init__(self, landmark_pos, max_ep_len=max_ep_len):
        self.landmark_pos = landmark_pos
        self.max_ep_len = max_ep_len
        for i in range(n_landmarks):
            for j in range(n_landmarks):
                if i != j:
                    dist = (self.landmark_pos[i] - self.landmark_pos[j]).norm()
                    assert dist > constraint_landmark_dist, (
                        dist, constraint_landmark_dist)

    @property
    def obs_dim(self):
        return 2 + n_landmarks * 2

    @property
    def act_dim(self):
        return 2

    @torch.no_grad()
    def reset(self):
        self._episode_length = 0
        self.pos = torch.zeros(2, dtype=torch.float32, device=device)
        return torch.cat([self.pos, self.landmark_pos.flatten()])

    @torch.no_grad()
    def step(self, action):
        action = action.clip(-1, 1).to(device)
        self.pos += action * delta_t
        self.pos.clamp_(-world_size, world_size)
        self._episode_length += 1
        obs = torch.cat([self.pos, self.landmark_pos.flatten()])
        reward = 0
        done = False
        info = dict(nearest_landmark=torch.zeros(n_landmarks))
        for i, p in enumerate(self.landmark_pos):
            if (self.pos - p).norm() < eps + agent_size + landmark_size:
                reward = 1
                done = True
                info['nearest_landmark'][i] = 1
                break
        if self._episode_length > self.max_ep_len:
            done = True
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        fig = plt.figure(figsize=(6, 6))
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.set_xlim([-world_size, world_size])
        ax.set_ylim([-world_size, world_size])

        width, height = fig.get_size_inches() * fig.get_dpi()

        landmark_circles = [
            plt.Circle((pos[0].cpu().numpy(), pos[1].cpu().numpy()),
                       radius=landmark_size,
                       linewidth=0,
                       color='r') for pos in self.landmark_pos
        ]
        [ax.add_artist(c) for c in landmark_circles]

        agent_circle = plt.Circle(
            (self.pos[0].cpu().numpy(), self.pos[1].cpu().numpy()),
            radius=agent_size,
            linewidth=0,
            color='b')
        ax.add_artist(agent_circle)

        canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(int(height), int(width), 3).copy()
        plt.close(fig)
        return image


class PPOBuffer:
    """Modified from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py#L12."""

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros((size, obs_dim),
                                   dtype=torch.float32,
                                   device=device)
        self.act_buf = torch.zeros((size, act_dim),
                                   dtype=torch.float32,
                                   device=device)
        self.adv_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.val_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = self.path_start_idx
        if end_idx is None:
            end_idx = self.ptr

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat([
            self.rew_buf[path_slice],
            torch.Tensor([last_val]).to(self.rew_buf)
        ])
        vals = torch.cat([
            self.val_buf[path_slice],
            torch.Tensor([last_val]).to(self.val_buf)
        ])

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas,
                                                   self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return dict(obs=self.obs_buf,
                    act=self.act_buf,
                    ret=self.ret_buf,
                    adv=self.adv_buf,
                    logp=self.logp_buf)


class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim],
                          activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class ActorCritic(nn.Module):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_sizes=(64, 64),
                 activation=nn.Tanh):
        super().__init__()
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation)
        self.v = Critic(obs_dim, hidden_sizes, activation)

    @torch.no_grad()
    def step(self, obs):
        pi = self.pi._distribution(obs)
        a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]


def pbt_ppo(rank,
            world_size,
            landmark_pos,
            lagrangian_lambda,
            constraint_type='min',
            seed=0,
            steps_per_epoch=4000,
            epochs=80,
            gamma=0.997,
            clip_ratio=0.2,
            pi_lr=3e-4,
            vf_lr=1e-3,
            train_pi_iters=10,
            train_v_iters=10,
            lam=0.97,
            save_freq=10,
            log_freq=5):
    """Modified from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py#L88."""
    assert constraint_type in ['min', 'pairwise']
    torch.distributed.init_process_group(distributed_backend,
                                         rank=rank,
                                         world_size=world_size,
                                         init_method='tcp://localhost:12345')

    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed + rank * 1000)
    torch.cuda.manual_seed_all(seed + rank * 1000)
    np.random.seed(seed + rank * 1000)

    # Instantiate environment
    env = NaviEnv(landmark_pos.to(device))
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Create actor-critic module
    ac = ActorCritic(env.obs_dim, env.act_dim).to(device)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data[
            'logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = clipped.float().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    def update(step):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.info(f"Rank {rank} update epochs {epoch + 1}/{epochs}...")
        log_info(dict(LossPi=pi_l_old,
                      LossV=v_l_old,
                      Entropy=ent,
                      ClipFrac=cf,
                      DeltaLossPi=(loss_pi.item() - pi_l_old),
                      DeltaLossV=(loss_v.item() - v_l_old)),
                 step=step,
                 rank=rank)

    trajectories = collections.deque(maxlen=n_save_trajs)
    cur_traj = []

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    cur_traj.append(o)

    logger.info(f"Rank {rank} training has started...")
    env_info = collections.defaultdict(lambda: 0)
    nearest_landmark = 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        final_states = []
        bootstrap_values = []
        on_final_step = torch.zeros(local_steps_per_epoch, dtype=torch.int32)
        for t in range(local_steps_per_epoch):
            print(rank, t)
            a, v, logp = ac.step(o)

            next_o, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1
            if d:
                final_states.append(env.pos.clone())
                on_final_step[t] = 1

            # save and log
            buf.store(o, a, r, v, logp)

            # Update obs (critical!)
            o = next_o

            cur_traj.append(o)
            if d:
                if ep_ret > 0:
                    trajectories.append(torch.stack(cur_traj))
                cur_traj = []

            terminal = d
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                # buf.finish_path(v)
                bootstrap_values.append(v)
                if terminal:
                    env_info['ep_ret'] += ep_ret
                    env_info['ep_len'] += ep_len
                    env_info['ep_cnt'] += 1
                    nearest_landmark += info['nearest_landmark'].cpu().numpy()
                o, ep_ret, ep_len = env.reset(), 0, 0
                cur_traj.append(o)

        print(1)
        torch.distributed.barrier()

        # gather final states and compute pairwise distance
        final_states = torch.stack(final_states)

        N = torch.ones(1) * final_states.shape[0]
        N_to_gather = [torch.zeros_like(N) for _ in range(n_landmarks)]
        torch.distributed.all_gather(N_to_gather, N)
        N_max = max([int(n) for n in N_to_gather])

        final_states_to_gather = [
            torch.zeros(N_max, 2).to(final_states) for _ in range(n_landmarks)
        ]
        final_states = torch.cat(
            [final_states, torch.zeros(int(N_max - N), 2)], dim=0)
        torch.distributed.all_gather(final_states_to_gather, final_states)
        final_states_to_gather = [
            s[:int(n)] for s, n in zip(final_states_to_gather, N_to_gather)
        ]

        if constraint_type == 'min':
            min_dist = torch.inf
            min_id_tuple = None
            for i in range(n_landmarks):
                for j in range(i + 1, n_landmarks):
                    x1 = final_states_to_gather[i]
                    x2 = final_states_to_gather[j]
                    dist = l2_dist(x1, x2).mean()
                    if dist < min_dist:
                        min_dist = dist
                        min_id_tuple = (i, j)

            i, j = min_id_tuple
            if rank == i or rank == j:
                other_rank = j if rank == i else i
                x1 = final_states_to_gather[rank]
                x2 = final_states_to_gather[other_rank]
                dist = l2_dist(x1, x2).mean(1)
                assert on_final_step.sum() == dist.shape[0], (
                    on_final_step.sum(), dist.shape)
                if on_final_step[-1]:
                    assert on_final_step.sum() == len(bootstrap_values)
                else:
                    assert on_final_step.sum() == len(bootstrap_values) - 1, (
                        len(bootstrap_values), on_final_step)
                idx = 0
                path_start_t = 0
                for t in range(local_steps_per_epoch):
                    if on_final_step[t]:
                        buf.rew_buf[t] += dist[idx] * lagrangian_lambda
                        buf.finish_path(bootstrap_values[idx], path_start_t,
                                        t + 1)
                        idx += 1
                        path_start_t = t + 1
                if not on_final_step[-1]:
                    assert idx == len(bootstrap_values) - 1, (
                        idx, len(bootstrap_values))
                    buf.finish_path(bootstrap_values[-1], path_start_t,
                                    local_steps_per_epoch)
            else:
                idx = 0
                path_start_t = 0
                for t in range(local_steps_per_epoch):
                    if on_final_step[t]:
                        buf.finish_path(bootstrap_values[idx], path_start_t,
                                        t + 1)
                        idx += 1
                        path_start_t = t + 1
                if not on_final_step[-1]:
                    assert idx == len(bootstrap_values) - 1, (
                        idx, len(bootstrap_values))
                    buf.finish_path(bootstrap_values[-1], path_start_t,
                                    local_steps_per_epoch)
        elif constraint_type == 'pairwise':
            subset_ll = torch.zeros(n_landmarks - 1)
            idx1 = idx2 = 0
            for i in range(n_landmarks):
                for j in range(i + 1, n_landmarks):
                    if rank in (i, j):
                        subset_ll[idx1] = lagrangian_lambda[idx2]
                        idx1 += 1
                    idx2 += 1
            assert idx1 == n_landmarks - 1 and idx2 == int(
                n_landmarks * (n_landmarks - 1) / 2)

            dist = []
            x1 = final_states_to_gather[rank]
            for i in range(n_landmarks):
                if i != rank:
                    x2 = final_states_to_gather[i]
                    dist.append(l2_dist(x1, x2).mean(1))
            dist = torch.stack(dist, -1)

            idx = 0
            path_start_t = 0
            for t in range(local_steps_per_epoch):
                if on_final_step[t]:
                    buf.rew_buf[t] += (dist[idx] * subset_ll).sum()
                    buf.finish_path(bootstrap_values[idx], path_start_t, t + 1)
                    idx += 1
                    path_start_t = t + 1
            if not on_final_step[-1]:
                assert idx == len(bootstrap_values) - 1, (
                    idx, len(bootstrap_values))
                buf.finish_path(bootstrap_values[-1], path_start_t,
                                local_steps_per_epoch)

        # Perform PPO update!
        update(epoch)

        if constraint_type == 'min':
            if rank == min(i, j):
                lagrangian_lambda += lagrangian_lr * (constraint_agent_dist -
                                                      min_dist)
                lagrangian_lambda.clamp_(0, ll_max)
                logger.info(
                    f"Lagrange Lambda: {lagrangian_lambda.cpu().numpy()}, "
                    f"constraint indices {(i, j)},"
                    f" min distance {min_dist}, "
                    f"constraint distance {constraint_agent_dist}.")
        elif constraint_type == 'pairwise':
            dist = torch.zeros(int(n_landmarks * (n_landmarks - 1) / 2))
            idx = 0
            for i in range(n_landmarks):
                for j in range(i + 1, n_landmarks):
                    x1 = final_states_to_gather[i]
                    x2 = final_states_to_gather[j]
                    dist[idx] = l2_dist(x1, x2).mean()
                    idx += 1
            assert idx == int(n_landmarks * (n_landmarks - 1) / 2)
            dist_to_gather = [
                torch.zeros_like(dist) for _ in range(n_landmarks)
            ]
            torch.distributed.all_gather(dist_to_gather, dist)
            for _dist in dist_to_gather:
                assert (dist == _dist).all(), (dist, _dist)
            if rank == 0:
                lagrangian_lambda += lagrangian_lr * (constraint_agent_dist -
                                                      dist)
                lagrangian_lambda.clamp_(0, ll_max)
                logger.info(
                    f"Lagrange Lambda: {lagrangian_lambda.cpu().numpy()}, "
                    f"pairwise distance {dist.cpu().numpy()}, "
                    f"constraint distance {constraint_agent_dist}.")

        if (epoch + 1) % log_freq == 0 or epoch == epochs - 1:
            if env_info['ep_cnt'] > 0:
                env_info['ep_ret'] /= env_info['ep_cnt']
                env_info['ep_len'] /= env_info['ep_cnt']
                log_info(env_info, step=epoch, rank=rank)
                logger.info(
                    f"Rank {rank} Nearest landmark {nearest_landmark / env_info['ep_cnt']}."
                )
                env_info = collections.defaultdict(lambda: 0)
                nearest_landmark = 0

        if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
            torch.save(
                list(trajectories),
                os.path.join(pbt_traj_dir, f"traj_pbt{rank}_{epoch + 1}.pt"))
            trajectories = collections.deque(maxlen=n_save_trajs)


def pbt_main(args, landmark_pos):
    if args.pbt_constraint_type == 'min':
        lagrangian_lambda = torch.zeros(()).share_memory_()
    elif args.pbt_constraint_type == 'pairwise':
        lagrangian_lambda = torch.zeros(
            int(n_landmarks * (n_landmarks - 1) / 2)).share_memory_()
    if not os.path.exists(pbt_traj_dir):
        os.makedirs(pbt_traj_dir)
    procs = [
        mp.Process(target=pbt_ppo,
                   kwargs=dict(rank=i,
                               seed=args.seed,
                               constraint_type=args.pbt_constraint_type,
                               lagrangian_lambda=lagrangian_lambda,
                               world_size=n_landmarks,
                               landmark_pos=landmark_pos))
        for i in range(n_landmarks)
    ]
    for p in procs:
        p.start()
    while any([p.is_alive() for p in procs]):
        [p.join(timeout=1.0) for p in procs]


def iter_ppo(iteration,
             archive_states,
             landmark_pos,
             seed=0,
             steps_per_epoch=4000,
             epochs=80,
             gamma=0.997,
             clip_ratio=0.2,
             pi_lr=3e-4,
             vf_lr=1e-3,
             train_pi_iters=10,
             train_v_iters=10,
             lam=0.97,
             save_freq=10,
             log_freq=5):
    """Modified from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py#L88."""
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed + iteration * 1000)
    torch.cuda.manual_seed_all(seed + iteration * 1000)
    np.random.seed(seed + iteration * 1000)

    assert iteration == len(archive_states)
    if iteration > 0:
        lagrangian_lambda = torch.zeros(iteration).to(device)

    # Instantiate environment
    env = NaviEnv(landmark_pos.to(device))
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Create actor-critic module
    ac = ActorCritic(env.obs_dim, env.act_dim).to(device)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data[
            'logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = clipped.float().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    def update(step):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.info(f"Update epochs {epoch + 1}/{epochs}...")
        log_info(dict(LossPi=pi_l_old,
                      LossV=v_l_old,
                      Entropy=ent,
                      ClipFrac=cf,
                      DeltaLossPi=(loss_pi.item() - pi_l_old),
                      DeltaLossV=(loss_v.item() - v_l_old)),
                 step=step)

    trajectories = collections.deque(maxlen=n_save_trajs)
    cur_traj = []

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    cur_traj.append(o)

    logger.info(f"Training has started...")
    env_info = collections.defaultdict(lambda: 0)
    nearest_landmark = 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        state_norms = []
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(o)

            next_o, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1
            if d:
                final_state = env.pos.clone()
                if len(archive_states) > 0:
                    intrinsic_rewards = torch.zeros(
                        len(archive_states),
                        dtype=torch.float32).to(final_state)
                    for j, archive_state in enumerate(archive_states):
                        intrinsic_rewards[j] = (archive_state -
                                                final_state).norm(
                                                    dim=1).mean()
                    state_norms.append(intrinsic_rewards)
                    intrinsic_rewards = (intrinsic_rewards *
                                         lagrangian_lambda).sum()
                    r += intrinsic_rewards

            # save and log
            buf.store(o, a, r, v, logp)

            # Update obs (critical!)
            o = next_o

            cur_traj.append(o)
            if d:
                if ep_ret > 0:
                    trajectories.append(torch.stack(cur_traj))
                cur_traj = []

            terminal = d
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                # TODO: multiple value heads
                if epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    env_info['ep_ret'] += ep_ret
                    env_info['ep_len'] += ep_len
                    env_info['ep_cnt'] += 1
                    nearest_landmark += info['nearest_landmark'].cpu().numpy()
                o, ep_ret, ep_len = env.reset(), 0, 0
                cur_traj.append(o)

        # Perform PPO update!
        update(epoch)

        # update lagrangian coefficients
        if len(archive_states) > 0:
            lagrangian_lambda += lagrangian_lr * (
                constraint_agent_dist - torch.stack(state_norms).mean(0))
            lagrangian_lambda.clamp_(0, ll_max)
            logger.info(f"Lagrange Lambda: {lagrangian_lambda.cpu().numpy()}")
            logger.info(
                f"Average intrinsic rewards: {torch.stack(state_norms).mean(0)}"
            )

        if (epoch + 1) % log_freq == 0 or epoch == epochs - 1:
            if env_info['ep_cnt'] > 0:
                env_info['ep_ret'] /= env_info['ep_cnt']
                env_info['ep_len'] /= env_info['ep_cnt']
                log_info(env_info, step=epoch)
                logger.info(
                    f"Nearest landmark {nearest_landmark / env_info['ep_cnt']}."
                )
                env_info = collections.defaultdict(lambda: 0)
                nearest_landmark = 0

        if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
            torch.save(
                list(trajectories),
                os.path.join(iter_traj_dir,
                             f"traj_iter{iteration}_{epoch + 1}.pt"))
            trajectories = collections.deque(maxlen=n_save_trajs)

    final_states = []
    nearest_landmark = 0
    env_info = collections.defaultdict(lambda: 0)
    for t in range(local_steps_per_epoch):
        with torch.no_grad():
            a = ac.pi.mu_net(o)
        next_o, r, d, info = env.step(a)

        o = next_o

        if d:
            env_info['ep_ret'] += ep_ret
            env_info['ep_len'] += ep_len
            env_info['ep_cnt'] += 1
            nearest_landmark += info['nearest_landmark'].cpu().numpy()
            final_states.append(env.pos.clone())
            o = env.reset()
    logger.info(
        f"Eval Episode Return {env_info['episode_return'] / env_info['ep_cnt']}, "
        f"Eval Episode Length {env_info['ep_len'] / env_info['ep_cnt']}, "
        f"Eval Episode Cnt {env_info['episode_cnt']}, "
        f"Nearest landmark {nearest_landmark / env_info['ep_cnt']}.")
    return torch.stack(final_states)


def iter_main(args, landmark_pos):
    if not os.path.exists(iter_traj_dir):
        os.makedirs(iter_traj_dir)
    archive_states = []
    for iteration in range(n_landmarks):
        logger.info(f"Iteration {iteration} has started...")
        state = iter_ppo(iteration,
                         seed=args.seed,
                         landmark_pos=landmark_pos,
                         archive_states=archive_states)
        logger.info(f"Iteration {iteration} ends, "
                    f"agent final state {state.mean(0).cpu().numpy()}...")
        archive_states.append(state)


# logging
logger = logging.getLogger(args.paradigm)
foramtter = logging.Formatter(
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(foramtter)
logger.addHandler(sh)

fh = logging.FileHandler(f'{args.paradigm}.txt', mode='w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(foramtter)
logger.addHandler(fh)

if __name__ == "__main__":
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    landmark_pos = generate_landmark_pos()
    env = NaviEnv(landmark_pos.to(device))
    env.reset()
    env.pos = torch.zeros(2)
    image = env.render()
    cv2.imwrite(args.render_name, image)
    if args.paradigm == 'pbt':
        pbt_main(args, landmark_pos)
    elif args.paradigm == 'iter':
        iter_main(args, landmark_pos)
    elif args.paradigm == 'both':
        iter_main(args, landmark_pos)
        pbt_main(args, landmark_pos)
    elif args.paradigm == 'render':
        pass
    else:
        raise NotImplementedError("A paradigm must be given (pbt or iter).")
