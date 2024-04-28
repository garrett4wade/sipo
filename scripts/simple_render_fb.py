import numpy as np
from environment.football.simple_renderer import render_from_observation
import torch
import os

n_iterations = 4
scenario = '3v1'
for algo in ['sipo-rbf']:
    for scenario in ['3v1']:
        seed = 1
        iteration = 1
        batch_index = 2

        path_root = f"/sipo_archive/checkpoints/fb_{scenario}/{algo}/seed{seed}/"
        traj = torch.load(
            os.path.join(path_root, f"win_traj{iteration}.pt"))
        obs = traj.obs.obs[:100, batch_index, 0]
        print(traj.obs.ball_owned_player[:100, batch_index, 0, 0])
        print(traj.masks[:100, batch_index, 0, 0])
        render_from_observation(obs, "demo.gif")
        print("done")
