# Continuous forest environments

import numpy as np

import rlberry.spaces as spaces
from rlberry.envs.interface import Model


from forest_risk_rl.utils import (
    build_transition_matrix,
    make_grid_matrix,
    make_octo_grid_matrix,
)
from forest_risk_rl.risk_measures import (
    windthrow_risk_continuous,
    group_risk,
    diversity_risk,
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


class LinearQuadraticEnv(Model):
    """
    Linear Quadratic Environment (Linear dynamics, Quadratic rewards)

    Parameters
    ----------
    n_states : int
        Number of states.
    n_actions : int
        Number of actions.
    A : np.ndarray
        State transition matrix.
    B : np.ndarray
        Action transition matrix.
    M : np.ndarray
        State reward matrix.
    N : np.ndarray
        Action reward matrix.
    R : function
        Hazard and noise generator.
    high : float
        Upper bound of the observation space.
    low : float
        Lower bound of the observation space.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        A=None,
        B=None,
        M=None,
        N=None,
        R=None,
        high=1.0,
        low=-1.0,
    ):
        # Default values
        if A is None:
            A = np.eye(n_states)
        if B is None:
            B = np.ones((n_actions, n_states))
        if M is None:
            M = np.zeros((n_states, n_states))
        if N is None:
            N = np.zeros((n_actions, n_actions))
        if R is None:
            R = lambda state, action: np.zeros(n_states)
        self.n_states = n_states
        self.n_actions = n_actions
        self.A = A
        self.B = B
        self.M = M
        self.N = N
        self.R = R
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
        next_state = self.A @ state + self.B @ action + self.R(state, action)
        reward = (
            state.transpose() @ self.M @ state + action.transpose() @ self.N @ action
        )
        done = False
        info = {}
        return next_state, reward, done, info

    def reset(self):
        self.state = np.zeros(self.n_states)
        return self.state.copy()


class ForestLinearEnv(LinearQuadraticEnv):
    """
    Forest environment.

    Parameters
    ----------
    n_tree : int
        Number of trees.
    adjacency_matrix : np.ndarray
        Adjacency matrix of the forest.
    H : float
        Height parameter (Influence the asymptotic height of the forest).
    alpha : float
        Growth parameter (Influence the growth rate of the forest).
    beta : float
        Interaction parameter (Influence the interaction between trees).
    R : function
        Hazard and noise generator.
    """

    def __init__(
        self,
        n_tree=10,
        adjacency_matrix=None,
        H=20,
        alpha=0.5,
        beta=0.5,
        R=None,
    ):
        self.n_states = n_tree + 1
        self.n_actions = n_tree
        self.H = H
        self.alpha = alpha
        self.beta = beta
        self.n_tree = n_tree
        self.adjacency_matrix = adjacency_matrix.astype(bool)

        A = build_transition_matrix(adjacency_matrix, alpha, beta)
        B = -A @ np.vstack((np.eye(n_tree), np.zeros(n_tree)))
        M = np.zeros((self.n_states, self.n_states))
        N = np.eye(self.n_actions) / self.H**2
        super().__init__(
            self.n_states, self.n_actions, A, B, M, N, R=R, high=2 * H, low=0.0
        )

        self.action_space = spaces.MultiBinary(n_tree)

    def reset(self):
        self.state = np.random.uniform(0, self.H, self.n_states)
        self.state[-1] = self.H
        return self.state.copy()

    def sample(self, state, action):
        action_vector = self._make_action_vector(state, action)
        return super().sample(state, action_vector)

    def _make_action_vector(self, state, action):
        """
        Make action vector from action.
        """
        K = np.zeros((self.n_tree, self.n_tree))
        for i in range(len(action)):
            if action[i] == 1:
                K[i, i] = 1
        K = np.hstack((K, np.zeros((self.n_tree, 1))))
        return K @ state

    def compute_risks(self, state):
        """
        Compute the risk of a given state
        """
        windthrow_risk = np.zeros(self.n_tree)
        for i in range(self.n_tree):
            windthrow_risk[i] = windthrow_risk_continuous(
                state[:-1][self.adjacency_matrix[i]], lambda x: (self.H - x) / self.H
            )

        diversity_risk = np.zeros(self.n_tree)
        group_risk = np.zeros(self.n_tree)

        return windthrow_risk, diversity_risk, group_risk


class ForestLinearEnvCA(ForestLinearEnv):
    """
    Forest environment with continuous action (which are discretized by the environment)
    """

    def __init__(self, n_tree=10, adjacency_matrix=None, H=20, alpha=0.5, beta=0.5):
        super().__init__(n_tree, adjacency_matrix, H, alpha, beta)
        self.action_space = spaces.Box(0, 1, shape=(n_tree,))

    def sample(self, state, action):
        discrete_action = np.round(action).astype(int)
        action_vector = self._make_action_vector(state, discrete_action)
        return super().sample(state, action_vector)


class ForestWithStorms(ForestLinearEnv):
    """
    Forest environment with storms.

    Parameters
    ----------
    n_tree : int
        Number of trees.
    adjacency_matrix : np.ndarray
        Adjacency matrix of the forest.
    H : float
        Height parameter (Influence the asymptotic height of the forest).
    alpha : float
        Growth parameter (Influence the growth rate of the forest).
    beta : float
        Interaction parameter (Influence the interaction between trees).
    storm_prob : float
        Probability of a storm at each time step.
    max_degree : int
        Maximum degree of the forest.
    """

    def __init__(
        self,
        n_tree=10,
        adjacency_matrix=None,
        H=20,
        alpha=0.5,
        beta=0.5,
        storm_prob=0.1,
        storm_power=None,
    ):
        self.storm_prob = storm_prob
        if storm_power is None or not isinstance(storm_power, int):
            storm_power = np.sum(adjacency_matrix, axis=1).max()
        self.D = storm_power
        super().__init__(n_tree, adjacency_matrix, H, alpha, beta, R=self.random_storm)

    def random_storm(self, state, action):
        """
        Random storm generator.

        Parameters
        ----------
        state : np.ndarray
            State of the environment.
        action : np.ndarray
            Action of the agent.
        """
        R = np.zeros_like(action)
        if np.random.rand() < self.storm_prob:
            for i in range(self.n_tree):
                p = np.exp(
                    -np.sum(state[:-1][self.adjacency_matrix[i]]) / (self.H * self.D)
                )
                if np.random.rand() < p:
                    R[i] = 1
                    # print(i, state[i], p)
        K_prime = np.zeros((self.n_tree, self.n_tree))
        for i in range(len(action)):
            if R[i] == 1:
                K_prime[i, i] = 1
        return self.B @ K_prime @ (state[:-1] - action)


if __name__ == "__main__":
    env = ForestWithStorms(
        n_tree=9,
        adjacency_matrix=make_octo_grid_matrix(3, 3),
        H=20,
        alpha=0.2,
        beta=0.11,
        storm_prob=0.0,
        storm_power=2,
    )

    observation = env.reset()
    states = [observation]
    for i in range(50):
        action = [0] * 9
        if i % 10 == 0:
            action[0] = 1
        observation, reward, done, info = env.step(action)
        states.append(observation)

    plt.plot(states)
    plt.xlabel("Time step")
    plt.ylabel("Height")
    plt.title("Periodic cutting of a tree")
    plt.show()