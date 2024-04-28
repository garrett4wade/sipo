import numpy as np
import ot
import math

tau = [[(2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), (4, 5)],
       [(1, 2), (2, 2), (3, 2), (3, 3), (3, 4), (4, 4), (5, 4)],
       [(1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (4, 5)]]


def emd(idx1, idx2):
    cost = np.zeros((14, 14))
    for i, s1 in enumerate(tau[idx1] + tau[idx2]):
        for j, s2 in enumerate(tau[idx1] + tau[idx2]):
            cost[i, j] = math.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)
    # print(cost)
    T = ot.emd(np.concatenate([np.ones(7), np.zeros(7)]),
               np.concatenate([np.zeros(7), np.ones(7)]), cost)
    return (T * cost).sum()

print(emd(0, 1))
print(emd(0, 2))
print(emd(2, 0))
print(np.sqrt(((np.array(tau[0]).flatten() - np.array(tau[1]).flatten()).astype(np.float32)**2).sum()))
print(np.sqrt(((np.array(tau[0]).flatten() - np.array(tau[2]).flatten()).astype(np.float32)**2).sum()))
