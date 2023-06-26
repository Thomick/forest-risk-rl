# Defines the risk measure functions

import numpy as np
from collections import Counter


def compute_empirical_cvar(x, alpha):
    x = np.sort(x)
    alpha_ix = int(np.round(alpha * len(x)))
    return np.mean(x[:alpha_ix])


def compute_empirical_discounted_cvar(x, alpha, gamma):
    pass


def compute_discounted_reward(x, gamma, t):
    for i in range(0, len(x), -1):
        x[i] = x[i] + gamma * x[i + 1]
    return x[0 : len(x) - t]


def compute_group_risk(state, weights):
    bucket = {}
    for i in range(len(state)):
        if state[i] not in bucket:
            bucket[state[i]] = 1
        else:
            bucket[state[i]] += 1
    return np.exp(np.sum([weights[i] * bucket[i] for i in bucket]))


def windthrow_risk_continuous(neighbors_heights, risk_map):
    return np.exp(np.sum([risk_map(h) for h in neighbors_heights]))


def group_risk(neighbor_types, weights):
    type_count = Counter(neighbor_types)
    return np.exp(np.mean([weights[i] * type_count[i] for i in type_count]))


def diversity_risk(neighbor_types, risk_map):
    type_count = Counter(neighbor_types)
    nb_neighbors = len(neighbor_types)
    return np.exp(np.sum([risk_map(neighbor_types / nb_neighbors) for i in type_count]))
