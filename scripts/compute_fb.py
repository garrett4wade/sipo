from typing import Tuple, Optional
import os
import torch
import numpy as np
import itertools
import collections
import pandas

np.set_printoptions(precision=2)


def get_fb_behavior_emd_from_win_traj_batch(traj, eps=0.03) -> Optional[Tuple]:
    n_trajs = traj.masks.shape[1]
    if n_trajs < 1:
        return None
    tmp_behaviors = collections.defaultdict(lambda: 0)
    for batch_index in range(n_trajs):
        mask = traj.masks[:, batch_index, 0].squeeze(-1)
        ball_pos = traj.obs.obs[:, batch_index, 0, 88:90].unsqueeze(-2)
        player_pos = traj.obs.obs[:, batch_index, 0,
                                  2:2 * num_players + 2].view(
                                      -1, num_players, 2)
        player_ball_dist = (player_pos - ball_pos).norm(dim=-1)
        valid_mask = torch.logical_and(mask > 0,
                                       player_ball_dist.min(-1).values < eps)
        emd_duplicate = player_ball_dist.argmin(
            -1)[valid_mask].numpy().tolist()
        emd = tuple(x for x, _ in itertools.groupby(emd_duplicate))
        tmp_behaviors[emd] += 1
    return max(tmp_behaviors, key=tmp_behaviors.get)


n_iterations = 3

for map_name in ['ca_easy', '3v1', 'corner']:
    if map_name == '3v1':
        num_players = 3
    else:
        num_players = 10

    for algo in ['sipo-rbf']:
        data = []
        for seed in range(1, 7):
            traj_root = f"results/{algo}/{map_name}/check/{seed}/run1"
            flag = False
            behaviors = set()
            for iteration in range(n_iterations):
                try:
                    traj = torch.load(os.path.join(traj_root, f"iter{iteration}/win_traj.pt")
                    )
                except FileNotFoundError:
                    flag = True
                    break
                emd = get_fb_behavior_emd_from_win_traj_batch(traj)
                if emd is not None:
                    behaviors.add(emd)
            if not flag:
                data.append(len(behaviors))
        data = sorted(data)[-3:]
        print(algo, np.mean(data), np.std(data), data)