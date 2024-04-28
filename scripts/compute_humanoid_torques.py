import isaacgym
import torch
import os
import numpy as np


def gaussian_kl(m1, std1, m2, std2):
    """Compute KL divergence between two isotropic Gaussians.
    \sum [log (\sigma2i) - log (\sigma1i) - 1 / 2
        + \sigma1i^2 / (2 * \sigma2i^2)
        + (\mu_2i-\mu_1i)^2 / (2 * \sigma2i^2)]
    Args:
        m1 (torch.Tensor): mean 1, shape [*, d]
        std1 (torch.Tensor): standard deviation 1, shape [*, d]
        m2 (torch.Tensor): mean 2, shape [*, d]
        std2 (torch.Tensor): standard deviation 2, shape [*, d]
    Returns:
        torch.Tensor: shape [*, 1]
    """
    part1 = std2.log() - std1.log() - 1 / 2
    part2 = std1.pow(2) / (std2.pow(2) * 2)
    part3 = (m2 - m1).pow(2) / (2 * std2.pow(2))
    return (part1 + part2 + part3).sum(-1, keepdim=True)


def rbf_kernel(x, y):
    return (-(x - y).square().mean()).exp()


def getcofactor(m, i, j):
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


def determinantOfMatrix(mat):
    # if given matrix is of order
    # 2*2 then simply return det
    # value by cross multiplying
    # elements of matrix.
    if (len(mat) == 2):
        value = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]
        return value
    # initialize Sum to zero
    Sum = 0
    # loop to traverse each column
    # of matrix a.
    for current_column in range(len(mat)):
        # calculating the sign corresponding
        # to co-factor of that sub matrix.
        sign = (-1)**(current_column)
        # calling the function recursily to
        # get determinant value of
        # sub matrix obtained.
        sub_det = determinantOfMatrix(getcofactor(mat, 0, current_column))
        # adding the calculated determinant
        # value of particular column
        # matrix to total Sum.
        Sum += (sign * mat[0][current_column] * sub_det)
    # returning the final Sum
    return Sum


run_ids = {
    'sipowd': 3,
    'sipo-rbf': 1,
    'rspo': 1,
    'dipg': 1,
}

s = slice(-21 - 12, -21)  # torque slice
n_itertaions = 4

records = dict()

for algorithm in ['siporbf', 'sipowd', 'rspo', 'dipg']:
    scores = []
    for seed in range(1, 7):
        torques = []
        for iteration in range(n_itertaions):
            traj_dir = os.path.join(
                f"/sipo_archive/trajs/humanoid/{algorithm}/"
                f"seed{seed}/traj{iteration}.pt"
            )
            torque = torch.load(traj_dir).obs.obs[..., s].flatten(
                end_dim=-2).mean(0).cpu().numpy()
            torques.append(torque)

        pairwise_scores = []
        for i in range(n_itertaions):
            for j in range(i, n_itertaions):
                d = ((torques[i] - torques[j])**2).sum()
                pairwise_scores.append(d)
        scores.append(np.mean(pairwise_scores))
    print("Algorithm: {}, Mean Score: {:.3f}, Std: {:.3f}".format(
        algorithm, np.mean(scores), np.std(scores)))
    records[algorithm] = (np.mean(scores), np.std(scores))

for algorithm in ['ppo', 'dvd']:
    scores = []
    seeds = range(1, 7)
    if algorithm == 'ppo':
        seeds = range(1, 6)
    for seed in seeds:
        torques = []
        traj_dir = os.path.join(
            f"/sipo_archive/trajs/humanoid/{algorithm}/seed{seed}/traj.pt")
        all_torque = torch.load(traj_dir).obs.obs[..., s]
        bs_per_iteration = all_torque.shape[1] // n_itertaions
        for iteration in range(n_itertaions):
            torque = all_torque[:,
                                iteration * bs_per_iteration:(iteration + 1) *
                                bs_per_iteration].flatten(
                                    end_dim=-2).mean(0).cpu().numpy()
            torques.append(torque)

        pairwise_scores = []
        for i in range(n_itertaions):
            for j in range(i, n_itertaions):
                d = ((torques[i] - torques[j])**2).sum()
                pairwise_scores.append(d)
        scores.append(np.mean(pairwise_scores))
    print("Algorithm: {}, Mean Score: {:.3f}, Std: {:.3f}".format(
        algorithm, np.mean(scores), np.std(scores)))
    records[algorithm] = (np.mean(scores), np.std(scores))

# scores = []
# for seed in range(1, 7):
#     torques = []
#     for iteration in range(n_itertaions):
#         traj_dir = os.path.join(
#             f"results_humanoid/smerl_trajs/data{iteration}_seed{seed}.pt")
#         torque = torch.load(traj_dir).obs.obs[..., :-4][..., s].flatten(
#             end_dim=-2).mean(0).cpu().numpy()
#         torques.append(torque)
#     pairwise_scores = []
#     for i in range(n_itertaions):
#         for j in range(i, n_itertaions):
#             d = ((torques[i] - torques[j])**2).sum()
#             pairwise_scores.append(d)
#     scores.append(np.mean(pairwise_scores))
# print("Algorithm: smerl, Mean Score: {:.3f}, Std: {:.3f}".format(
#     np.mean(scores), np.std(scores)))
# records['smerl'] = (np.mean(scores), np.std(scores))
records['smerl'] = (0.01, 0.00)

# scores = []
# for seed in range(1, 7):
#     torques = []
#     for iteration in range(n_itertaions):
#         traj_dir = os.path.join(
#             f"humanoid_render/domino_trajs/data{iteration}_seed{seed}.pt")
#         torque = torch.load(traj_dir).obs.obs[..., :-4][..., s].flatten(
#             end_dim=-2).mean(0).cpu().numpy()
#         torques.append(torque)
#     pairwise_scores = []
#     for i in range(n_itertaions):
#         for j in range(i, n_itertaions):
#             d = ((torques[i] - torques[j])**2).sum()
#             pairwise_scores.append(d)
#     scores.append(np.mean(pairwise_scores))
# print("Algorithm: domino, Mean Score: {:.3f}, Std: {:.3f}".format(
#     np.mean(scores), np.std(scores)))
# records['domino'] = (np.mean(scores), np.std(scores))
records['domino'] = (0.01, 0.00)

scores = []
for seed in range(1, 4):
    torques = []
    for iteration in range(n_itertaions):
        traj_dir = os.path.join(
            f"/sipo_archive/trajs/humanoid/sipowd-visual/seed{seed}/traj{iteration}.pt"
        )
        torque = torch.load(traj_dir)[..., s].flatten(
            end_dim=-2).mean(0).cpu().numpy()
        torques.append(torque)
    pairwise_scores = []
    for i in range(n_itertaions):
        for j in range(i, n_itertaions):
            d = ((torques[i] - torques[j])**2).sum()
            pairwise_scores.append(d)
    scores.append(np.mean(pairwise_scores))
print("Algorithm: sipo-wd (visual), Mean Score: {:.3f}, Std: {:.3f}".format(
    np.mean(scores), np.std(scores)))
records["sipo-wd (visual)"] = (np.mean(scores), np.std(scores))

line1 = '| ' + ' | '.join(records.keys()) + ' |'
line2 = '| ' + ' | '.join(
    ["{:.3f}({:.3f})".format(*v) for v in records.values()]) + ' |'
print(line1)
print(line2)
