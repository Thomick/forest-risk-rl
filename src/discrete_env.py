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

    def __init__(self, n_tree=10, adjacency_matrix=None, H=20, a=1):
        if adjacency_matrix is None:
            adjacency_matrix = np.ones((n_tree, n_tree)) - np.eye(n_tree)
        self.n_states = n_tree
        self.n_actions = n_tree
        self.H = H
        self.a = a
        self.n_tree = n_tree
        self.adjacency_matrix = adjacency_matrix
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

        p_growth = np.zeros(self.n_states)
        for j in range(self.n_states):
            p_growth[j] = (
                np.exp((s[j] - np.mean(s[self.adjacency_matrix[j]])) / self.H)
                / np.exp(self.a)
                * (self.H - s[j])
                / self.H
            )
        p_growth.clip(0, 1, out=p_growth)
        for j in range(self.n_states):
            if np.random.rand() < p_growth[j]:
                s[j] = min(s[j] + 1, self.H)

        self.state = s.copy()
        done = False
        return self.state, reward, done, {}

    def _reward_from_height(self, height):
        return height**2


if __name__ == "__main__":
    row, col = 5, 5
    n_tree = row * col
    H = 20
    a = 1
    adjacency_matrix = make_grid_matrix(row, col).astype(int)
    env = ForestDiscreteEnv(n_tree=n_tree, adjacency_matrix=adjacency_matrix, H=H, a=a)
    observation = env.reset()
    env.state = np.zeros(n_tree)
    states = [observation]

    for i in range(100):
        action = [0] * env.n_actions
        observation, reward, done, info = env.step(action)
        states.append(observation)

    plt.plot(states)
    plt.figure()
    sns.heatmap(observation.reshape(row, col))
    plt.show()
