# Discrete forest environments

import rlberry.spaces as spaces
from rlberry.envs.interface import Model

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from utils import make_grid_matrix


class ForestDiscreteEnv(Model):
    """
    Forest environment with discrete states and actions.

    Parameters
    ----------
    n_tree : int
        Number of trees.
    adjacency_matrix : np.ndarray
        Adjacency matrix of the forest.
    H : int
        Maximum height of the trees.
    a : float
        Scaling parameter for the growth probability.
    """

    def __init__(
        self, n_tree=10, adjacency_matrix=None, H=20, a=1, storm_power=4, p_storm=0
    ):
        if adjacency_matrix is None:
            adjacency_matrix = np.ones((n_tree, n_tree)) - np.eye(n_tree)
        self.n_states = n_tree
        self.n_actions = n_tree
        self.H = H
        self.a = a
        self.D = storm_power
        self.n_tree = n_tree
        self.adjacency_matrix = adjacency_matrix.astype(bool)
        self.p_storm = p_storm
        self.action_space = spaces.MultiBinary(n_tree)
        self.observation_space = spaces.MultiDiscrete([H + 1] * n_tree)

        self.reset()

    def reset(self):
        self.state = np.random.randint(0, self.H, self.n_tree)
        return self.state.copy()

    def step(self, action):
        reward = 0
        s = self.state.copy()

        for j in range(self.n_actions):
            if action[j] == 1:
                reward += self._reward_from_height(s[j])
                s[j] = 0

        p_windthrow = self._compute_p_windthrow(s)
        for j in range(self.n_states):
            if np.random.rand() < p_windthrow[j]:
                s[j] = 0

        p_growth = self._compute_p_growth(s)
        for j in range(self.n_states):
            if np.random.rand() < p_growth[j]:
                s[j] = min(s[j] + 1, self.H)

        self.state = s.copy()
        done = False
        return self.state, reward, done, {}

    def _reward_from_height(self, height):
        return height**2

    def _compute_p_growth(self, state):
        p_growth = np.zeros(self.n_states)
        for j in range(self.n_states):
            p_growth[j] = (
                np.exp((state[j] - np.mean(state[self.adjacency_matrix[j]])) / self.H)
                / np.exp(self.a)
                * (self.H - state[j])
                / self.H
            )
        p_growth.clip(0, 1, out=p_growth)
        return p_growth

    def _compute_p_windthrow(self, state):
        p_windthrow = np.zeros(self.n_states)
        if np.random.rand() < self.p_storm:
            for j in range(self.n_states):
                p_windthrow[j] = np.exp(
                    -np.sum(state[self.adjacency_matrix[j]]) / (self.H * self.D)
                )
            p_windthrow.clip(0, 1, out=p_windthrow)
        return p_windthrow


if __name__ == "__main__":
    row, col = 5, 5
    n_tree = row * col
    H = 20
    a = 1
    adjacency_matrix = make_grid_matrix(row, col).astype(int)
    env = ForestDiscreteEnv(
        n_tree=n_tree, adjacency_matrix=adjacency_matrix, H=H, a=a, p_storm=0.05
    )
    observation = env.reset()
    env.state = np.zeros(n_tree)
    # env.state[[2 * i for i in range((n_tree + 1) // 2)]] = H
    states = [observation]

    for i in range(100):
        action = [0] * env.n_actions
        observation, reward, done, info = env.step(action)
        states.append(observation)

    plt.plot(states)
    plt.figure()
    sns.heatmap(observation.reshape(row, col))
    plt.show()
