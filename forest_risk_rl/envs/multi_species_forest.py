from forest_risk_rl.envs.linear_dynamic_env import ForestLinearEnv, ForestWithStorms
from forest_risk_rl.utils import (
    build_transition_matrix,
    make_octo_grid_matrix,
    make_grid_matrix,
)
import numpy as np
import gym
import gym.spaces as spaces


class MultiSpeciesForest(ForestLinearEnv):
    """
    Multi-species forest environment.

    Parameters
    ----------
    n_tree : int
        Number of trees.
    adjacency_matrix : np.ndarray
        Adjacency matrix of the forest.
    H : float
        Height parameter (Influence the asymptotic height of the forest).
    alpha : np.ndarray
        Growth parameters for each type of trees (Influence the growth rate of the forest).
    beta : float
        Interaction parameter for each type of trees (Influence the interaction between trees).
    R : function
        Hazard and noise generator.
    """

    def __init__(
        self,
        nb_tree,
        adjacency_matrix,
        H=20,
        alpha_list=[0.0, 0.1],
        beta_list=[0.0, 0.05],
        R=None,
    ):
        self.alpha_list = np.array(alpha_list)
        self.beta_list = np.array(beta_list)
        super().__init__(nb_tree, adjacency_matrix, H, alpha_list[0], beta_list[0], R)
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(self.low, self.high, shape=(nb_tree + 1,)),
                spaces.MultiDiscrete([2] * nb_tree),
            )
        )
        self.action_space = spaces.MultiDiscrete([3] * nb_tree)

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        return (next_state, self.tree_types), reward, done, info

    def sample(self, state, action):
        for i in range(self.n_tree):
            if action[i] != 0:
                self.tree_types[i] = action[i] - 1
        self._update_transition_matrix()
        action = (action > 0).astype(int)
        return super().sample(state, action)

    def reset(self):
        self.tree_types = np.random.randint(0, 2, size=self.n_tree).astype(int)
        self._update_transition_matrix()
        return super().reset()

    def _update_transition_matrix(self):
        self.A = build_transition_matrix(
            self.adjacency_matrix,
            self.alpha_list[self.tree_types],
            self.beta_list[self.tree_types],
        )
        self.B = -self.A @ np.vstack((np.eye(self.n_tree), np.zeros(self.n_tree)))


if __name__ == "__main__":
    env = MultiSpeciesForest(
        nb_tree=9,
        adjacency_matrix=make_octo_grid_matrix(3, 3),
        alpha_list=[0.0, 0.1],
        beta_list=[0.0, 0.05],
    )
    env.reset()
    env.step(np.zeros(9).astype(int))
    observation, reward, done, info = env.step(np.zeros(9).astype(int))
    print(observation)
    print(env.A)
