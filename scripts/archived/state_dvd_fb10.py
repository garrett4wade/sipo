import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import os, socket
import pandas as pd
from numpy import nan

data = {"map_names": ['3v1']}
for algo in ['sipo-rbf', 'sipo-wd', 'rspo', 'dipg']:
    data[algo + "_mean"] = np.zeros(3)
    data[algo + "_std"] = np.ones(3)
    if algo == 'rspo':
        data[algo + "_mean"][2] = np.nan
        data[algo + "_std"][2] = np.nan
    elif algo == 'dvd' or algo == 'smerl':
        data[algo + "_mean"][1:] = np.nan
        data[algo + "_std"][1:] = np.nan

np.set_printoptions(precision=3)

num_stack = 4
n_iterations = 10
sigma = 0.15


def feature_selector(cent_state, map_name):
    return torch.cat([cent_state[..., :-4], cent_state[..., -3:-1]], -1)
    # n_player = 3 if map_name == '3v1' or map_name == "3vs1" else 10
    # ep_len, bs = cent_state.shape[:2]

    # ball_posvel = cent_state[..., -6:].flatten(end_dim=1)
    # ball_xy = ball_posvel[..., :2].unsqueeze(-2)
    # player_posvel = cent_state[..., :-6].reshape(-1, n_player, 4)
    # player_xy = player_posvel[..., :2]

    # player_idx = (player_xy - ball_xy).pow(2).sum(-1).argmin(-1)

    # batch_indices = torch.arange(ep_len * bs)
    # x = torch.cat([
    #     player_xy[batch_indices, player_idx], ball_xy[..., :2].squeeze(-2),
    # ], -1)
    # return x.reshape(ep_len, bs, 4)
    # ep_len = cent_state.shape[0]
    # if map_name == '3v1' or map_name == "3vs1":
    #     return torch.cat([cent_state[..., :-4], cent_state[..., -3:-1]], -1)
    # elif map_name == 'ca_easy' or map_name == "counterattack":
    #     cent_state = cent_state[..., 4 * 6:]
    #     return torch.cat([cent_state[..., :-4], cent_state[..., -3:-1]], -1)
    # elif map_name == 'corner':
    #     cent_state = cent_state[..., 4:]
    #     return torch.cat([
    #         cent_state[..., :12], cent_state[..., 16:-4], cent_state[...,
    #                                                                  -3:-1]
    #     ], -1)


def dist(x, y):
    return x.pow(2).sum(-1, keepdim=True) + y.pow(2).sum(-1) - 2 * x @ y.T


for map_idx, map_name in enumerate(["3v1"]):
    cent_state_dim = 18
    algos = ['sipo-rbf', 'rspo', 'dipg']
    for algo in algos:
        pds = []
        for seed in range(1, 4):
            samples = [
                torch.load(
                    f"/sipo_archive/trajs/football/{map_name}/{algo}/seed{seed}/archive_data{i}.traj"
                ) for i in range(4)
            ]
            if algo == 'sipo-rbf':
                run_dir = "run1" if seed != 1 else "run4"
                samples += [
                    torch.load(
                        f"/fast_mappo/results/{map_name}/check/{seed}/{run_dir}/iter{j}/models/data.traj"
                    ) for j in range(4, 10)
                ]
            else:
                samples += [
                    torch.load(
                        f"/fast_mappo/results/{algo}/{map_name}/check/{seed}/run1/iter{j}/models/data.traj"
                    ) for j in range(4, 10)
                ]

            cent_states = []
            masks = []
            for sample in samples:
                bs = sample.masks.shape[1]
                # remove agent dim
                bs = sample.masks.shape[1]
                # remove agent dim
                cent_state = sample.obs.cent_state[..., 0, :]
                mask = sample.masks[..., 0, :]
                # deal with frame stack
                if cent_state.shape[-1] != cent_state_dim:
                    assert cent_state.shape[
                        -1] % num_stack == 0, cent_state.shape
                    state_dim = cent_state.shape[-1] // num_stack
                    cent_state = cent_state[..., -state_dim:]
                    assert cent_state.shape[-1] == cent_state_dim, (
                        cent_state_dim, cent_state.shape[-1])
                for j in range(bs):
                    if (mask[:, j] == 0).sum() == 0:
                        mask[:, j] = 0
                    else:
                        ep2start_idx = int(
                            (mask[:, j] == 0).flatten().nonzero()[0])
                        # print(ep2start_idx, mask[:, j])
                        mask[ep2start_idx:, j] = 0
                        if sample.rewards[:ep2start_idx, j, 0].sum() < 1:
                            mask[:, j] = 0

                # remove z-axis of the ball
                cent_states.append(feature_selector(cent_state, map_name))
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
                        # assert mask.sum() > 0, mask.sum()
                        dist = (dist * mask).sum() / (mask.sum() + 1e-5)
                        # dist = dist.mean()
                        # if i == 0 and j == 1:
                        #     print(dist)
                        # ep_len = min(x.shape[0], y.shape[0])
                        # print(((cent_states[i] - cent_states[j])**2).sum(-1).mean())
                        pd[i, j] = pd[j, i] = np.exp(-dist / 2 / sigma**2)
            # print(pd)
            pd = np.linalg.det(pd)
            if pd > 0:
                pds.append(pd)
        data[algo + "_mean"][map_idx] = np.mean(pds)
        data[algo + "_std"][map_idx] = np.std(pds)
        # if len(pds) > 0:
        #     print(map_name, algo, np.mean(pds), np.std(pds), pds)
        # else:
        #     print(map_name, algo, "N/A")
    print("------------------")

map_agent_registry = {
    # evn_name: (left, right, game_length, total env steps)
    # keeper is not included in controllable players
    "3vs1": (3, 1, 400, int(25e6)),
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


for map_idx, map_name in enumerate(["3vs1"]):
    cent_state_dim = 18
    pds = []
    for seed in range(1, 4):
        samples = [
            torch.load(
                f"/sipo_archive/trajs/football/sipowd_traj/football_{map_name}_seed{seed}_iter{ref_i}.traj"
            ) for ref_i in range(n_iterations)
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
                    ep2start_idx = int((mask[:,
                                             j] == 0).flatten().nonzero()[0])
                    # print(ep2start_idx, mask[:, j])
                    mask[ep2start_idx:, j] = 0
                if sample.rewards[:ep2start_idx, j, 0].sum() < 1:
                    mask[:, j] = 0

            # remove z-axis of the ball
            cent_states.append(feature_selector(cent_state, map_name))
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
    data["sipo-wd_mean"][map_idx] = np.mean(pds)
    data["sipo-wd_std"][map_idx] = np.std(pds)
print("------------------")

for k, v in data.items():
    if isinstance(v, np.ndarray):
        data[k] = v.tolist()

print(data)
