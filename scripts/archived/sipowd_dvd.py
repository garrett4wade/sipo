import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import os, socket

np.set_printoptions(precision=3)

num_stack = 4
n_iterations = 4


def dist(x, y):
    return x.pow(2).sum(-1, keepdim=True) + y.pow(2).sum(-1) - 2 * x @ y.T


map_agent_registry = {
    # evn_name: (left, right, game_length, total env steps)
    # keeper is not included in controllable players
    "3vs1": (3, 1, 400, int(25e6)),
    "corner": (10, 10, 400, int(50e6)),
    "counterattack": (10, 10, 400, int(25e6)),
}


def make_cent_state(obs, map_name):
    n_l_players = map_agent_registry[map_name][0]
    n_r_players = map_agent_registry[map_name][1]
    cent_state = np.concatenate([
        obs[..., 2:2 + n_l_players * 2], obs[..., 24:24 + n_l_players * 2],
        obs[..., 88:88 + 6]
    ], -1)
    assert (cent_state[..., 0, :] == cent_state[..., -1, :]).all()
    return torch.from_numpy(cent_state)


for map_name in ["3vs1", "counterattack", "corner"]:
    sigma = 0.4
    cent_state_dim = 18 if map_name == "3vs1" else 46
    pds = []
    for seed in range(1, 4):
        samples = [
            torch.load(f"/sipo_archive/trajs/football/sipowd_traj/football_{map_name}_seed{seed}_iter{ref_i}.traj")
            for ref_i in range(n_iterations)
        ]

        cent_states = []
        masks = []
        for sample in samples:
            bs = sample.masks.shape[1]
            # remove agent dim
            obs_dim = sample.obs.obs.shape[-1]
            assert obs_dim == 115, obs_dim
            cent_state = make_cent_state(sample.obs.obs, map_name)[..., 0, :]
            mask = sample.masks[..., 0, :]
            for j in range(bs):
                if (mask[:, j] == 0).sum() == 0:
                    mask[:, j] = 0
                else:
                    ep2start_idx = int((mask[:, j] == 0).flatten().nonzero()[0])
                    # print(ep2start_idx, mask[:, j])
                    mask[ep2start_idx:, j] = 0
                if sample.rewards[:ep2start_idx, j, 0].sum() < 1:
                    mask[:, j] = 0

            # remove z-axis of the ball
            cent_states.append(
                torch.cat([cent_state[..., :-4], cent_state[..., -3:-1]], -1))
            masks.append(mask)
        # print([x.shape for x in cent_states])

        pd = np.zeros((n_iterations, n_iterations))
        for i in range(len(cent_states)):
            for j in range(i, len(cent_states)):
                if i == j:
                    pd[i, j] = 1
                else:
                    x = cent_states[i].flatten(end_dim=1)
                    y = cent_states[j].flatten(end_dim=1)
                    mx = masks[i].flatten(end_dim=1)
                    my = masks[j].flatten(end_dim=1)
                    mask = mx * my.squeeze()
                    dist = x.pow(2).sum(-1, keepdim=True) + y.pow(2).sum(
                        -1) - 2 * x @ y.T  # [N, M]
                    # if i == 0 and j == 1:
                    #     print(x, y)
                    #     print(dist * mask)
                    dist = (dist * mask).sum() / (mask.sum() + 1e-5)
                    # dist = dist.mean()
                    # if i == 0 and j == 1:
                    #     print(dist)
                    # ep_len = min(x.shape[0], y.shape[0])
                    # print(((cent_states[i] - cent_states[j])**2).sum(-1).mean())
                    pd[i, j] = pd[j, i] = np.exp(-dist / 2 / sigma**2)
        pd = np.linalg.det(pd)
        if pd > 0:
            pds.append(pd)
    if len(pds) > 0:
        print(map_name, np.mean(pds), np.std(pds), pds)
    else:
        print(map_name, "N/A")
print("------------------")
