import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import os, socket
import pandas as pd
from numpy import nan
from sklearn.neighbors import NearestNeighbors

data = {"scenarios": ['3v1', 'CA', 'Corner']}
for algo in ['sipo-rbf', 'sipo-wd', 'rspo', 'dipg', 'dvd', 'smerl', 'pg']:
    data[algo] = ['-'] * 3

np.set_printoptions(precision=3)

num_stack = 4
n_iterations = 4


def feature_selector(cent_state, scenario):
    return torch.cat([cent_state[..., :-6]], -1)


def get_pd_from_cent_states(cent_states, masks):
    cent_state_dim = cent_states[0].shape[-1]
    cent_states = [x.reshape(-1, cent_state_dim) for x in cent_states]
    masks = [m.reshape(-1, 1) for m in masks]
    cent_state = torch.cat(cent_states, 0).numpy()
    mask = torch.cat(masks, 0).numpy()
    X = cent_state[np.tile(mask, (1, cent_state.shape[-1])) > 0].reshape(
        -1, cent_state.shape[-1])
    nbrs = NearestNeighbors(n_neighbors=12).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return np.log(distances[..., -1] + 1).mean(0)


for map_idx, map_name in enumerate(["3v1", "ca", 'corner']):
    cent_state_dim = 18 if map_name == "3v1" else 46
    algos = ['rspo', 'dipg']
    if map_name == '3v1':
        algos += ['dvd', 'smerl']
    elif map_name == 'corner':
        algos.remove('rspo')
    for algo_name in algos:
        pds = []
        for seed in range(1, 4):
            archive_dir = f"/sipo_archive/win_trajs/fb_{map_name}/{algo_name}/seed{seed}"
            # archive_dir = f"2m_vs_1z_rspo_data_seed{seed}"
            samples = [
                torch.load(os.path.join(archive_dir, f"iter{ref_i}.pt"))
                for ref_i in range(n_iterations)
            ]

            cent_states = []
            masks = []
            for sample in samples:
                ep_len = sample.masks[..., 0, 0].sum(0)
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
                # remove z-axis of the ball
                cent_states.append(feature_selector(cent_state, map_name))
                masks.append(mask)
            # print([x.shape for x in cent_states])

            pd = get_pd_from_cent_states(cent_states, masks)
            pds.append(pd)
        data[algo_name][map_idx] = f"{np.mean(pds):.3f}({np.std(pds):.3f})"
        if len(pds) > 0:
            print(map_name, algo_name, np.mean(pds), np.std(pds), pds)
        else:
            print(map_name, algo_name, "N/A")

num_stack = 4
n_iterations = 4

map_agent_registry = {
    # evn_name: (left, right, game_length, total env steps)
    # keeper is not included in controllable players
    "3v1": (3, 1, 400, int(25e6)),
    "corner": (10, 10, 400, int(50e6)),
    "ca": (10, 10, 400, int(25e6)),
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


for map_idx, map_name in enumerate(["3v1", "ca", "corner"]):
    cent_state_dim = 18 if map_name == "3v1" else 46
    pds = []
    for seed in range(1, 4):
        samples = [
            torch.load(
                f"/sipo_archive/trajs/fb_{map_name}/sipowd/seed{seed}/archive_data{ref_i}.traj"
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

        pd = get_pd_from_cent_states(cent_states, masks)
        pds.append(pd)
    data["sipo-wd"][map_idx] = f"{np.mean(pds):.3f}({np.std(pds):.3f})"
    # data["sipo-wd_mean"][map_idx] = np.mean(pds)
    # data["sipo-wd_std"][map_idx] = np.std(pds)

for algo in ['sipo-rbf', 'pg']:
    for map_idx, map_name in enumerate(["3v1", "ca", "corner"]):
        cent_state_dim = 18 if map_name == "3v1" else 46
        pds = []
        for seed in range(1, 4):
            if algo == 'pg':
                samples = [
                    torch.load(
                        f"/sipo_archive/win_trajs/fb_{map_name}/sipo-rbf/seed{seed}/iter0.pt"
                    )
                ]
                for iteration in range(1, 4):
                    traj = torch.load(
                        f"/sipo_archive/win_trajs/fb_{map_name}/pg/seed{seed}/iter{iteration}.pt"
                    )
                    samples.append(traj)
            else:
                samples = [
                    torch.load(
                        f"/sipo_archive/win_trajs/fb_{map_name}/sipo-rbf/seed{seed}/iter{iteration}.pt"
                    ) for iteration in range(4)
                ]

            cent_states = []
            masks = []
            for sample in samples:
                bs = sample.masks.shape[1]
                # remove agent dim
                obs_dim = sample.obs.obs.shape[-1]
                assert obs_dim == 115, obs_dim
                cent_state = make_cent_state(sample.obs.obs, map_name)[...,
                                                                       0, :]
                mask = sample.masks[..., 0, :]

                # remove z-axis of the ball
                cent_states.append(feature_selector(cent_state, map_name))
                masks.append(mask)
            # print([x.shape for x in cent_states])

            pd = get_pd_from_cent_states(cent_states, masks)
            pds.append(pd)
        data[algo][map_idx] = f"{np.mean(pds):.3f}({np.std(pds):.3f})"
        # data[f"{algo}_mean"][map_idx] = np.mean(pds)
        # data[f"{algo}_std"][map_idx] = np.std(pds)

# for k, v in data.items():
#     if isinstance(v, np.ndarray):
#         data[k] = v.tolist()

print(data)

for i in range(3):
    max_value, max_name = -1e10, None
    for k, v in data.items():
        if k.endswith('_mean') and v[i] > max_value:
            max_value = v[i]
            max_name = k

import pandas
df = pandas.DataFrame(data)
header = list([h.upper() for h in df.keys()])
header[0] = ""

latex_str = df.to_latex(float_format="%.3f",
                        index=False,
                        caption='State entropy in GRF.',
                        label='tab:ent-fb',
                        na_rep='-',
                        column_format='c' * len(header),
                        header=header)
# latex_str = df.to_markdown()
print("----------------")
print(latex_str)
