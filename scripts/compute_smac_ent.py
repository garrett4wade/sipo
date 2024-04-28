import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import os, socket
import pandas
from sklearn.neighbors import NearestNeighbors

np.set_printoptions(precision=3)

num_stack = 4
n_iterations = 4

all_data = {"algorithm": [], "2m1z": [], "2c64zg": []}


def state_entropy_from_cent_states(cent_states, masks):
    # print([x.shape for x in cent_states])
    cent_state = torch.cat(cent_states, 0).numpy()
    mask = torch.cat(masks, 0).numpy()
    X = cent_state[np.tile(mask, (1, cent_state.shape[-1])) > 0].reshape(
        -1, cent_state.shape[-1])
    nbrs = NearestNeighbors(n_neighbors=12).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return np.log(distances[..., -1] + 1).mean(0)


algos = ['sipo-rbf', 'sipo-wd', 'rspo', 'dipg', 'smerl', 'dvd']

for map_name in ['2m1z', "2c64zg"]:
    cent_state_dim = 10 if map_name == "2c64zg" else 8
    sigma = 0.4 if map_name == '2m1z' else 0.4
    for algo_name in algos:
        pds = []
        seeds = range(1, 7)
        for seed in seeds:
            samples = []
            archive_dir = f"/sipo_archive/win_trajs/smac_{map_name}/{algo_name}/seed{seed}"
            if not os.path.exists(archive_dir):
                # print(f"archive dir not exist: {archive_dir}")
                continue
            for ref_i in range(n_iterations):
                sample_dir = os.path.join(archive_dir, f"iter{ref_i}.pt")
                if not os.path.exists(sample_dir):
                    # print(f"sample not exist: {sample_dir}")
                    continue
                sample = torch.load(sample_dir)
                if sample.masks.sum() > 0:
                    samples.append(torch.load(sample_dir))

            cent_states = []
            masks = []
            for sample in samples:
                # remove agent dim
                cent_state = sample.obs.cent_state[..., 0, :]
                mask = sample.active_masks  # [T, B, A, 1]

                # deal with frame stack
                if cent_state.shape[-1] != cent_state_dim:
                    assert cent_state.shape[-1] % num_stack == 0
                    state_dim = cent_state.shape[-1] // num_stack
                    cent_state = cent_state[..., -state_dim:]
                    assert cent_state.shape[-1] == cent_state_dim, (
                        cent_state_dim, cent_state.shape[-1])

                cent_state = cent_state.view(*cent_state.shape[:-1], 2,
                                             cent_state_dim // 2)
                cent_state = cent_state[:, :, :, :]

                cent_states.append(cent_state.flatten(end_dim=-2))
                masks.append(mask[:, :].reshape(-1, 1))

            pds.append(state_entropy_from_cent_states(cent_states, masks))
        while len(pds) > 5:
            pds.remove(min(pds))
        print(map_name, algo_name, np.mean(pds), np.std(pds), pds)
        all_data[map_name].append(
            f"{np.mean(pds).item():.3f}({np.std(pds).item():.3f})")
        if algo_name not in all_data['algorithm']:
            if algo_name == 'sipo-rbf':
                if 'sipo(ours)' not in all_data['algorithm']:
                    all_data['algorithm'].append('sipo(ours)')
            else:
                all_data['algorithm'].append(algo_name)

print(all_data)
df = pandas.DataFrame(all_data)
header = list([h.upper() for h in df.keys()])
header[0] = ""

latex_str = df.to_latex(float_format="%.3f",
                        index=False,
                        caption='Population diversity in GRF.',
                        label='tab:state-dvd-fb',
                        na_rep='-',
                        column_format='c' * len(header),
                        header=header)
# latex_str = df.to_markdown()
print(latex_str)