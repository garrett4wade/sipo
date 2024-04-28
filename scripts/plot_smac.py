import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os, socket

plt.tight_layout()

map_info = {"2c_vs_64zg": (32, 32, 28, 28), "2m_vs_1z": (32, 32, 28, 28)}
# if map_name not in map_info:
#     from environment.smac.smac_env_ import StarCraft2Env
#     if socket.gethostname().startswith("frl"):
#         os.environ["SC2PATH"] = "/local/pkgs/StarCraftII"
#     example_env = StarCraft2Env(map_name=map_name)
#     example_env.reset()
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#     print(example_env.map_x, example_env.map_y, example_env.max_distance_x,
#           example_env.max_distance_y)
#     print(example_env.get_cent_state().shape)
#     print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#     map_info[map_name] = (example_env.map_x, example_env.map_y,
#                           example_env.max_distance_x,
#                           example_env.max_distance_y)
#     del example_env

algo = 'sipo-rbf'

levels = 10
num_agents = 2
num_stack = 4
fontsize = 18

for map_name in ['2m_vs_1z', '2c_vs_64zg']:

    seed = 2 if map_name == '2c_vs_64zg' else 3
    sigma = 2.0 if map_name == '2c_vs_64zg' else 1.8
    cent_state_dim = 10 if map_name == '2c_vs_64zg' else 8

    total_Zs = []

    map_x, map_y, map_distance_x, map_distance_y = map_info[map_name]
    center_x = map_x / 2
    center_y = map_y / 2

    plot_X, plot_Y = np.linspace(0, map_x, 100), np.linspace(0, map_y, 100)
    plot_X, plot_Y = np.meshgrid(plot_X, plot_Y)

    for ref_i in range(4):
        sample = torch.load(
            f"/home/fw/sipo_archive/trajs/smac/{map_name}/{algo}/seed{seed}/archive_data{ref_i}.traj"
        )

        # remove agent dim
        cent_state = sample.obs.cent_state[..., 0, :]

        # deal with frame stack
        if cent_state.shape[-1] != cent_state_dim:
            assert cent_state.shape[-1] % num_stack == 0
            state_dim = cent_state.shape[-1] // num_stack
            cent_state = cent_state[..., -state_dim:]
        assert cent_state.shape[-1] == cent_state_dim

        cent_state = cent_state.flatten(end_dim=1)
        cent_state = cent_state.view(cent_state.shape[0], num_agents, -1)

        total_Z = torch.zeros(plot_X.shape, dtype=torch.float32)

        for agent_id in range(num_agents):
            pos = cent_state[..., agent_id, 2:4]
            pos_x = pos[..., 0:1].view(-1, 1) * map_distance_x + center_x
            pos_y = pos[..., 1:2].view(-1, 1) * map_distance_y + center_y

            X = torch.linspace(0, map_x, 100).reshape(-1, 1)
            Y = torch.linspace(0, map_y, 100).reshape(-1, 1)

            dist_x = (X.pow(2) + pos_x.pow(2).squeeze() -
                      2 * X @ pos_x.T).T  # [N, 100]
            dist_y = (Y.pow(2) + pos_y.pow(2).squeeze() -
                      2 * Y @ pos_y.T).T  # [N, 100]
            Z = (-(dist_x.unsqueeze(-1) + dist_y.unsqueeze(-2)) / 2 /
                 sigma**2).exp().mean(0)
            total_Z += Z

            # plt.rc("figure", figsize=(8, 3))
            # plt.contourf(plot_X, plot_Y, Z.numpy().T, levels=10)
            # plt.ylim(0, 20)
            # plt.savefig(f'test_{ref_i}_agent{agent_id}.png')
        total_Z = (total_Z + 1e-5).sqrt()
        total_Zs.append(total_Z)
    # max_z = max([x.max() for x in total_Zs])
    # min_z = min([x.min() for x in total_Zs])
    total_Zs = [(total_Z - total_Z.min()) / (total_Z.max() - total_Z.min())
                for total_Z in total_Zs]

    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(12, 3))
    for i, (ax, total_Z) in enumerate(zip(axs, total_Zs)):
        CS = ax.contourf(plot_X,
                         plot_Y,
                         total_Z.numpy().T,
                         levels=levels,
                         cmap=plt.cm.cividis)
        ax.contour(plot_X,
                   plot_Y,
                   total_Z.numpy().T,
                   levels=levels,
                   cmap=plt.cm.jet)
        ax.set_title(f"policy {i+1}", fontsize=fontsize)
        if map_name == '2m_vs_1z':
            ax.set_xlim((4, 26))
            ax.set_ylim((4, 26))
        elif map_name == '2c_vs_64zg':
            # ax.set_xlim((4, 26))
            # ax.set_ylim((3, 23))
            pass
        # CS = ax2.contourf(X, Y, Z, 10, cmap=plt.cm.bone, origin=origin)

        # ax2.set_title('Nonsense (3 masked regions)')
        # ax2.set_xlabel('word length anomaly')
        # ax2.set_ylabel('sentence length anomaly')

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(CS)
    # fig.set_size_inches(6, 3)
    cbar.ax.set_ylabel('density', fontsize=fontsize)
    cbar.ax.set_yticks(np.linspace(0.0, 1.0, 11))
    ax.set_xlabel("x position", fontsize=fontsize)
    ax.set_ylabel("y position", fontsize=fontsize)
    # plt.rc("figure", figsize=(8, 3))
    plt.savefig(f'{map_name}_heatmap_{algo}.png', bbox_inches='tight')
