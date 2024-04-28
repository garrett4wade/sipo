from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import PatchCollection
import argparse
import collections
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

cmap = lambda x: plt.cm.jet(x)
colors = plt.cm.jet(np.linspace(0, 1, 4))

iter_traj_dir = "valid_iter_trajs/"
pbt_traj_dir = "valid_pbt_trajs/"
# landmark_pos = torch.load("navi_env_pos.pt")

landmark_size = 0.10
agent_size = 0.05
max_traj_len = 20

sigma = 0.23
agent_sigma = 0.3

landmark_pos = torch.tensor([[0.5928, -0.2108], [-0.0028, 0.5543],
                             [0.8943, 0.4214], [-0.1528, -0.3473]])
iter_converged_order = [0, 2, 3, 1]


def plot_traj(policy_trajs):
    fig = plt.figure(figsize=(6, 6))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.set_xlim([-0.5, 1.1])
    ax.set_ylim([-0.7, 0.9])
    # ax.axis('off')
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)

    width, height = fig.get_size_inches() * fig.get_dpi()

    # agent_circle = plt.Circle(
    #     (self.pos[0].cpu().numpy(), self.pos[1].cpu().numpy()),
    #     radius=agent_size,
    #     linewidth=0,
    #     color='b')
    # ax.add_artist(agent_circle)
    for policy_id, trajs in enumerate(policy_trajs):
        for traj in trajs:
            xy = traj[:max_traj_len, :2].cpu().numpy()
            ax.plot(xy[:, 0], xy[:, 1], color=colors[policy_id])

    landmark_circles = [
        plt.Circle((pos[0].cpu().numpy(), pos[1].cpu().numpy()),
                   radius=landmark_size,
                   linewidth=0,
                   color='r') for pos in landmark_pos
    ]
    [ax.add_artist(c) for c in landmark_circles]

    canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3).copy()
    plt.close(fig)
    return image


# for n_iterations in range(1, 5):
#     trajs = [
#         torch.load(os.path.join(iter_traj_dir, f"traj_iter{i}_70.pt"))
#         for i in range(n_iterations)
#     ]
#     # print(test_traj)
#     image = plot_traj(trajs)
#     cv2.imwrite(f"iter_result_{n_iterations}.png", image)

# a = torch.load("traj_iter0_10.pt")
# print(len(a))

# n_iterations = 3
# for epoch in range(10, 90, 10):
#     trajs = [
#         torch.load(os.path.join(iter_traj_dir, f"traj_iter{i}_{70 if i <=1 else epoch}.pt"))
#         for i in range(n_iterations)
#     ]
#     # print(test_traj)
#     image = plot_traj(trajs)
#     cv2.imwrite(f"iter_training_{epoch}.png", image)

# for epoch in range(10, 90, 10):
#     trajs = [
#         torch.load(os.path.join(pbt_traj_dir, f"traj_pbt{i}_{epoch}.pt"))
#         for i in range(4)
#     ]
#     # print(test_traj)
#     image = plot_traj(trajs)
#     cv2.imwrite(f"pbt_result_{epoch}.png", image)

if False:
    plt.tight_layout()
    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(12, 3))
    all_trajs = [
        torch.load(os.path.join(iter_traj_dir, f"traj_iter{i}_70.pt"))
        for i in range(4)
    ]
    final_states = []
    for trajs in all_trajs:
        final_states.append(
            torch.stack([traj[-1, :2] for traj in trajs]).mean(0))
    final_states = [final_states[idx] for idx in iter_converged_order]

    delta_Z = torch.zeros(100, 100)

    for final_state, (iteration, ax) in zip(final_states, enumerate(axs)):
        # all_trajs = [
        #     torch.load(os.path.join(iter_traj_dir, f"traj_iter0_{k}.pt"))
        #     for k in [10, 20]
        # ]
        # final_states = []
        # for trajs in all_trajs:
        #     final_states.append(torch.stack([traj[-1, :2] for traj in trajs]).mean(0))
        # final_states = torch.stack([torch.zeros_like(final_states[0])] +
        #                         final_states).cpu().numpy()
        ax.set_xlim([-0.5, 1.1])
        ax.set_ylim([-0.7, 0.9])
        # ax.axis('off')
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        # change all spines
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(4)

        X = torch.linspace(-0.5, 1.1, 100).view(-1, 1)
        Y = torch.linspace(-0.7, 0.9, 100).view(-1, 1)

        pos_x = landmark_pos[:, 0:1]
        pos_y = landmark_pos[:, 1:2]
        dist_x = (X.pow(2) + pos_x.pow(2).squeeze() -
                  2 * X @ pos_x.T).T  # [4, 100]
        dist_y = (Y.pow(2) + pos_y.pow(2).squeeze() -
                  2 * Y @ pos_y.T).T  # [4, 100]

        Z = (-(dist_x.unsqueeze(-1) + dist_y.unsqueeze(-2)) /
             (2 * sigma**2)).exp().mean(0).T

        X, Y = np.meshgrid(X.flatten(), Y.flatten())

        ax.contour(X, Y, Z + delta_Z)

        # add delta Z
        X = torch.linspace(-0.5, 1.1, 100).view(-1, 1)
        Y = torch.linspace(-0.7, 0.9, 100).view(-1, 1)

        pos_x = final_state[0].view(1, 1).cpu()
        pos_y = final_state[1].view(1, 1).cpu()
        dist_x = (X.pow(2) + pos_x.pow(2).squeeze() -
                  2 * X @ pos_x.T).T  # [4, 100]
        dist_y = (Y.pow(2) + pos_y.pow(2).squeeze() -
                  2 * Y @ pos_y.T).T  # [4, 100]
        delta_Z += (Z.max() - (-(dist_x.unsqueeze(-1) + dist_y.unsqueeze(-2)) /
                               (2 * agent_sigma**2)).exp().mean(0).T) * 0.26
        # ax.scatter(pos_x.flatten(), pos_y.flatten())

    fig.savefig('test.png', bbox_inches='tight')

plt.tight_layout()
fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(12, 3))
epoch_final_states = [None]
for epoch in [5, 15, 25]:
    all_trajs = [
        torch.load(os.path.join("valid_pbt_trajs2", f"traj_pbt{i}_{epoch}.pt"))
        for i in range(4)
    ]
    final_states = []
    for trajs in all_trajs:
        final_states.append(
            torch.stack([traj[-1, :2] for traj in trajs]).mean(0))
    epoch_final_states.append(final_states)

for final_states, (iteration, ax) in zip(epoch_final_states, enumerate(axs)):
    # all_trajs = [
    #     torch.load(os.path.join(iter_traj_dir, f"traj_iter0_{k}.pt"))
    #     for k in [10, 20]
    # ]
    # final_states = []
    # for trajs in all_trajs:
    #     final_states.append(torch.stack([traj[-1, :2] for traj in trajs]).mean(0))
    # final_states = torch.stack([torch.zeros_like(final_states[0])] +
    #                         final_states).cpu().numpy()
    ax.set_xlim([-0.5, 1.1])
    ax.set_ylim([-0.7, 0.9])
    # ax.axis('off')
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)

    X = torch.linspace(-0.5, 1.1, 100).view(-1, 1)
    Y = torch.linspace(-0.7, 0.9, 100).view(-1, 1)

    pos_x = landmark_pos[:, 0:1]
    pos_y = landmark_pos[:, 1:2]
    dist_x = (X.pow(2) + pos_x.pow(2).squeeze() -
              2 * X @ pos_x.T).T  # [4, 100]
    dist_y = (Y.pow(2) + pos_y.pow(2).squeeze() -
              2 * Y @ pos_y.T).T  # [4, 100]

    Z = (-(dist_x.unsqueeze(-1) + dist_y.unsqueeze(-2)) /
         (2 * sigma**2)).exp().mean(0).T

    delta_Z = torch.zeros(100, 100)

    if iteration > 0:
        for final_state in final_states:
            # add delta Z
            X = torch.linspace(-0.5, 1.1, 100).view(-1, 1)
            Y = torch.linspace(-0.7, 0.9, 100).view(-1, 1)

            pos_x = final_state[0].view(1, 1).cpu()
            pos_y = final_state[1].view(1, 1).cpu()
            dist_x = (X.pow(2) + pos_x.pow(2).squeeze() -
                      2 * X @ pos_x.T).T  # [4, 100]
            dist_y = (Y.pow(2) + pos_y.pow(2).squeeze() -
                      2 * Y @ pos_y.T).T  # [4, 100]
            delta_Z += (Z.max() -
                        (-(dist_x.unsqueeze(-1) + dist_y.unsqueeze(-2)) /
                         (2 * 0.15**2)).exp().mean(0).T) * 0.05
            # ax.scatter(pos_x.flatten(), pos_y.flatten())

    X, Y = np.meshgrid(X.flatten(), Y.flatten())
    ax.contour(X, Y, Z + delta_Z)

fig.savefig('test.png')