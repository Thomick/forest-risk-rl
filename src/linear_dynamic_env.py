import numpy as np

import rlberry.spaces as spaces
from rlberry.envs.interface import Model

from utils import build_transition_matrix, make_grid_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


class LinearDynamic(Model):
    """
    Linear dynamic environment.
    """

    def __init__(self, n_states=10, n_actions=2, A=None, B=None, high=1.0, low=-1.0):
        if A is None:
            A = np.eye(n_states)
        if B is None:
            B = np.ones((n_actions, n_states, n_states))
        self.n_states = n_states
        self.n_actions = n_actions
        self.A = A
        self.B = B
        self.low = low
        self.high = high

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(self.low, self.high, shape=(n_states,))

        self.reset()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        next_state, reward, done, info = self.sample(self.state, action)
        self.state = next_state.copy()
        return next_state, reward, done, info

    def sample(self, state, action):
        next_state = self.A @ state + self.B[action] @ state
        reward = state[action]
        done = False
        info = {}
        return next_state, reward, done, info

    def reset(self):
        self.state = np.zeros(self.n_states)
        return self.state.copy()


class ForestLinearEnv(LinearDynamic):
    def __init__(self, n_tree=10, adjacency_matrix=None, H=20, alpha=0.5, beta=0.5):
        n_states = n_tree + 1
        n_actions = 2
        self.H = H
        self.alpha = alpha
        self.beta = beta

        A = build_transition_matrix(adjacency_matrix, alpha, beta)
        B = np.zeros((2, n_states, n_actions))

        super().__init__(n_states, n_actions, A, B, high=2 * H, low=0.0)

        self.action_space = spaces.Tuple([spaces.Discrete(n_actions)] * n_tree)

    def reset(self):
        self.state = np.random.uniform(0, self.H, self.n_states)
        self.state[-1] = self.H
        return self.state.copy()

    def sample(self, state, action):
        B = np.zeros_like(self.A)
        for i in range(len(action)):
            if action[i] == 1:
                B[i, ...] = -self.A[i, ...]
        next_state = self.A @ state + B @ state
        reward = (B @ state).transpose() @ B @ state / self.H**2
        done = False
        info = {}
        return next_state, reward, done, info
