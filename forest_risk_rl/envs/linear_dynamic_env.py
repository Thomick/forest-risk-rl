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
from forest_risk_rl.simple_policies import ThresholdPolicy, CuttingAgePolicy

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
        next_state, reward, done, info = self.sample(self.state, action)
        self.state = next_state.copy()
        return next_state, reward, done, info

    def sample(self, state, action):
        next_state = self.A @ state + self.B @ action + self.R(state, action)
        next_state = np.clip(next_state, self.low, self.high)
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
    nb_tree : int
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
        nb_tree=10,
        adjacency_matrix=None,
        H=20,
        alpha=0.5,
        beta=0.5,
        R=None,
    ):
        self.n_states = nb_tree + 1
        self.n_actions = nb_tree
        self.H = H
        self.alpha = alpha
        self.beta = beta
        self.nb_tree = nb_tree
        self.adjacency_matrix = adjacency_matrix.astype(bool)

        A = build_transition_matrix(adjacency_matrix, alpha, beta)
        B = -A @ np.vstack((np.eye(nb_tree), np.zeros(nb_tree)))
        M = np.zeros((self.n_states, self.n_states))
        N = np.eye(self.n_actions) / self.H**2
        super().__init__(
            self.n_states, self.n_actions, A, B, M, N, R=R, high=2 * H, low=0.0
        )

        self.action_space = spaces.MultiBinary(nb_tree)
        self.step_since_reset = 0

    def reset(self):
        self.state = np.random.uniform(0, self.H, self.n_states)
        self.state[-1] = self.H
        self.step_since_reset = 0
        return self.state.copy()

    def sample(self, state, action):
        action_vector = self._make_action_vector(state, action)
        self.step_since_reset += 1
        return super().sample(state, action_vector)

    def _make_action_vector(self, state, action):
        """
        Make action vector from action.
        """
        K = np.zeros((self.nb_tree, self.nb_tree))
        for i in range(len(action)):
            if action[i] == 1:
                K[i, i] = 1
        K = np.hstack((K, np.zeros((self.nb_tree, 1))))
        return K @ state

    def compute_risks(self, state):
        """
        Compute the risk of a given state
        """
        windthrow_risk = np.zeros(self.nb_tree)
        for i in range(self.nb_tree):
            windthrow_risk[i] = windthrow_risk_continuous(
                state[:-1][self.adjacency_matrix[i]], lambda x: (self.H - x) / self.H
            )

        diversity_risk = np.zeros(self.nb_tree)
        group_risk = np.zeros(self.nb_tree)

        return windthrow_risk, diversity_risk, group_risk


class ForestLinearEnvCA(ForestLinearEnv):
    """
    Forest environment with continuous action (which are discretized by the environment)
    """

    def __init__(self, nb_tree=10, adjacency_matrix=None, H=20, alpha=0.5, beta=0.5):
        super().__init__(nb_tree, adjacency_matrix, H, alpha, beta)
        self.action_space = spaces.Box(0, 1, shape=(nb_tree,))

    def sample(self, state, action):
        discrete_action = np.round(action).astype(int)
        action_vector = self._make_action_vector(state, discrete_action)
        return super().sample(state, action_vector)


class ForestWithStorms(ForestLinearEnv):
    """
    Forest environment with storms.

    Parameters
    ----------
    nb_tree : int
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
    storm_sequence : list
        Sequence of storm occurences to be applied (if empty, storms are generated randomly). When arriving at the end of the sequence, the sequence is repeated.
    storm_mask : list or np.ndarray
        Mask describing which trees are affected by the storms. If None, all trees are affected. 0 means no storm, 1 means storm.
    """

    def __init__(
        self,
        nb_tree=10,
        adjacency_matrix=None,
        H=20,
        alpha=0.5,
        beta=0.5,
        storm_prob=0.1,
        storm_power=None,
        storm_sequence=[],
        storm_mask=None,
    ):
        self.storm_prob = storm_prob
        if storm_power is None or not isinstance(storm_power, int):
            storm_power = np.sum(adjacency_matrix, axis=1).max()
        self.D = storm_power
        self.storm_sequence = storm_sequence
        if storm_mask is None or len(storm_mask) != nb_tree:
            self.storm_mask = np.ones(nb_tree)
        else:
            self.storm_mask = np.array(storm_mask, dtype=int)
        super().__init__(
            nb_tree, adjacency_matrix, H, alpha, beta, R=self._generate_storm
        )

    def _generate_storm(self, state, action):
        """
        Storm generator (either random occurence or according to a fixed sequence set using storm_sequence parameter in the constructor)

        Parameters
        ----------
        state : np.ndarray
            State of the environment.
        action : np.ndarray
            Action of the agent.

        Returns
        -------
        R : np.ndarray
            Risk vector (used as noise in the environment).
        """
        R = np.zeros_like(action)

        storm_occurence = False
        if len(self.storm_sequence) > 0:
            storm_occurence = self.storm_sequence[
                self.step_since_reset % len(self.storm_sequence)
            ]
        elif np.random.rand() < self.storm_prob:
            storm_occurence = True

        if storm_occurence:
            for i in range(self.nb_tree):
                if self.storm_mask[i] == 0:
                    continue
                p = np.exp(
                    -np.sum(state[:-1][self.adjacency_matrix[i]]) / (self.H * self.D)
                )
                if np.random.rand() < p:
                    R[i] = 1
                    # print(i, state[i], p)
        K_prime = np.zeros((self.nb_tree, self.nb_tree))
        for i in range(len(action)):
            if R[i] == 1:
                K_prime[i, i] = 1
        return self.B @ K_prime @ (state[:-1] - action)


class ForestWithFires(ForestLinearEnv):
    """
    Forest environment with fires.

    Parameters
    ----------
    nb_tree : int
        Number of trees.
    adjacency_matrix : np.ndarray
        Adjacency matrix of the forest.
    H : float
        Height parameter (Influence the asymptotic height of the forest).
    alpha : float
        Growth parameter (Influence the growth rate of the forest).
    beta : float
        Interaction parameter (Influence the interaction between trees).
    fire_prob : float
        Probability of a fire at each time step.
    fire_duration : int
        Duration of the fire (number of propagation steps).
    """

    def __init__(
        self,
        nb_tree=10,
        adjacency_matrix=None,
        H=20,
        alpha=0.5,
        beta=0.5,
        fire_prob=0.1,
        fire_duration=3,
    ):
        self.fire_prob = fire_prob
        self.fire_duration = fire_duration
        super().__init__(
            nb_tree, adjacency_matrix, H, alpha, beta, R=self._generate_fire
        )

    def _generate_fire(self, state, action):
        """
        Fire generator

        Parameters
        ----------
        state : np.ndarray
            State of the environment.
        action : np.ndarray
            Action of the agent.

        Returns
        -------
        R : np.ndarray
            Fire vector (used as noise in the environment).
        """
        R = np.zeros_like(action)
        if np.random.rand() < self.fire_prob:
            starting_tree = np.random.randint(self.nb_tree)
            R[starting_tree] = 1
            for i in range(self.fire_duration):
                for j in range(self.nb_tree):
                    if R[j] == 1:
                        for k in range(self.nb_tree):
                            if self.adjacency_matrix[j, k] == 1 and R[k] == 0:
                                p_propagation = 0.5 / (
                                    1 + np.exp(-(state[j] - state[k]))
                                )
                                R[k] = np.random.rand() < p_propagation
            # print(R.reshape(5, 5))
        K_prime = np.zeros((self.nb_tree, self.nb_tree))
        for i in range(len(action)):
            if R[i] == 1:
                K_prime[i, i] = 1
        return self.B @ K_prime @ (state[:-1] - action)


def get_single_tree_env_class(base_class, tree_id, default_policy):
    """
    Returns a new environment class that only allows to act on a single tree in the given forest environment.

    Parameters
    ----------
    base_class : class
        Base environment class.
    tree_id : int
        Id of the tree to act on.
    default_policy : class or function
        Default policy to use for the other trees. Must be callable and take a state as input.

    Returns
    -------
    SingleTreeEnv : class
        New environment class.
    """

    class SingleTreeEnv(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tree_id = tree_id
            self.default_policy = default_policy
            self.action_space = spaces.Discrete(2)

        def step(self, action):
            full_action = self.default_policy(self.state)
            full_action[self.tree_id] = action
            return super().step(full_action)

    return SingleTreeEnv


def get_local_observation_env_class(base_class, tree_id, default_policy):
    """
    Returns a new environment class that only allows to observe the neighbors of a single tree (and only act on this tree) in the given forest environment.

    Parameters
    ----------
    base_class : class
        Base environment class.
    tree_id : int
        Id of the tree to act on.
    default_policy : class or function
        Default policy to use for the other trees. Must be callable and take a state as input.

    Returns
    -------
    LocalObservationEnv : class
        New environment class.
    """

    class LocalObservationEnv(
        get_single_tree_env_class(base_class, tree_id, default_policy)
    ):
        def __init__(self, *args, **kwargs):
            self.tree_id = tree_id
            super().__init__(*args, **kwargs)
            self.nb_neighbors = np.sum(self.adjacency_matrix[self.tree_id])
            self.observation_space = spaces.Box(
                self.low, self.high, shape=(self.nb_neighbors,)
            )

        def step(self, action):
            full_state, reward, done, info = super().step(action)
            return (
                full_state[:-1][self.adjacency_matrix[self.tree_id]],
                reward,
                done,
                info,
            )

        def reset(self):
            full_state = super().reset()
            return full_state[:-1][self.adjacency_matrix[self.tree_id]]

    return LocalObservationEnv


if __name__ == "__main__":
    """env = ForestWithStorms(
        nb_tree=9,
        adjacency_matrix=make_octo_grid_matrix(3, 3),
        H=20,
        alpha=0.2,
        beta=0.1,
        storm_prob=0.0,
        storm_power=2,
        storm_sequence=[False, False, False, False, True],
        storm_mask=[0, 0, 0, 0, 0, 0, 0, 0, 1],
    )"""

    """env = ForestWithFires(
        nb_tree=9,
        adjacency_matrix=make_octo_grid_matrix(3, 3),
        H=20,
        alpha=0.2,
        beta=0.1,
        fire_prob=0.05,
    )"""

    env = get_local_observation_env_class(ForestWithStorms, 4, ThresholdPolicy(9, 15))(
        nb_tree=9,
        adjacency_matrix=make_grid_matrix(3, 3),
        H=20,
        alpha=0.2,
        beta=0.1,
        storm_prob=0.0,
        storm_power=2,
    )

    observation = env.reset()
    states = [observation]
    total_reward = 0
    for i in range(50):
        action = 1 if i % 5 == 0 else 0
        observation, reward, done, info = env.step(action)
        states.append(observation)
        total_reward += reward
    print(total_reward)
    plt.plot(states)
    plt.xlabel("Time step")
    plt.ylabel("Height")
    plt.show()
