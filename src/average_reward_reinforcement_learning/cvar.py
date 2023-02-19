import numpy as np

from rlberry.agents import AgentWithSimplePolicy


def compute_empirical_cvar(x, alpha):
    x = np.sort(x)
    alpha_ix = int(np.round(alpha * len(x)))
    return np.mean(x[:alpha_ix])
