import numpy as np


def compute_empirical_cvar(x, alpha):
    x = np.sort(x)
    alpha_ix = int(np.round(alpha * len(x)))
    return np.mean(x[:alpha_ix])


def compute_group_risk(state, weights):
    bucket = {}
    for i in range(len(state)):
        if state[i] not in bucket:
            bucket[state[i]] = 1
        else:
            bucket[state[i]] += 1
    return np.exp(np.sum([weights[i] * bucket[i] for i in bucket]))
