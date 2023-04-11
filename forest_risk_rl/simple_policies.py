# Simple and expert policies

import numpy as np
import pandas as pd
from tqdm import tqdm


class SimplePolicy:
    """
    Abstract class for simple policies

    Methods
    -------
    __call__(state)
        Return the action to take given the state
    get_param()
        Return the parameter of the policy
    reset()
        Reset the internal state of the policy
    """

    name = "Simple policy"

    def __init__(self, nb_tree):
        self.nb_tree = nb_tree

    def __call__(self, state):
        raise NotImplementedError

    def get_param(self):
        raise NotImplementedError

    def reset(self):
        pass


class ThresholdPolicy(SimplePolicy):
    """
    A policy that always cut the tree if its height is above a threshold
    """

    name = "Threshold policy"

    def __init__(self, nb_tree, threshold):
        self.nb_tree = nb_tree
        self.threshold = threshold

    def __call__(self, state):
        return (np.array(state)[0 : self.nb_tree] >= self.threshold).astype(int)

    def get_param(self):
        return self.threshold


class CuttingAgePolicy(SimplePolicy):
    """
    A policy that always cut the tree if its age is above a threshold
    """

    name = "Cutting age policy"

    def __init__(self, nb_tree, cutting_age):
        self.cutting_age = cutting_age
        self.nb_tree = nb_tree
        self.reset()

    def __call__(self, state):
        self.age += 1
        to_cut = self.age >= self.cutting_age
        self.age[to_cut] = 0
        return to_cut.astype(int)

    def get_param(self):
        return self.cutting_age

    def reset(self):
        self.age = np.array([i % self.cutting_age for i in range(self.nb_tree)])


class FireBlockThreshold(ThresholdPolicy):
    """
    A policy that repeatedly cut the trees such that the fire does not spread to the entire forest.
    The forest is partitionned in 4 quadrants separated by areas with not trees.
    Inside each quadrant, the trees are cut if their height is above a threshold.
    """

    name = "Fire blocking policy"

    def __init__(self, nb_tree, threshold, nb_row, nb_col):
        super().__init__(nb_tree, threshold)
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.blocker_mask = np.zeros(
            (nb_row, nb_col)
        )  # The fire is restrained by partitionning the in smaller grids separated by area with not trees
        for i in range(nb_row):
            for j in range(nb_col):
                if i == nb_row // 2 or j == nb_col // 2:
                    self.blocker_mask[i, j] = 1

    def __call__(self, state):
        to_cut = super().__call__(state)
        return 1 - (1 - to_cut) * (1 - self.blocker_mask.flatten())

    def get_param(self):
        return self.threshold


class SynchronizedCutting(CuttingAgePolicy):
    """
    A policy that cut the trees synchronously when their age is above a threshold.
    """

    name = "Synchronized cutting policy"

    def __init__(self, nb_tree, cutting_age):
        super().__init__(nb_tree, cutting_age)
        self.age = 0

    def __call__(self, state):
        if self.age >= self.cutting_age:
            self.age = 0
            return np.ones(self.nb_tree)
        else:
            self.age += 1
            return np.zeros(self.nb_tree)

    def get_param(self):
        return self.cutting_age

    def reset(self):
        self.age = 0


class ExpertPolicy(SimplePolicy):
    """
    An expert policy
    """

    name = "Expert policy"

    def __init__(self, nb_tree, threshold, adjacency_matrix):
        self.nb_tree = nb_tree
        self.threshold = threshold
        self.adjacency_matrix = adjacency_matrix

    def __call__(self, state):
        index = np.zeros(self.nb_tree)
        for i in range(self.nb_tree):
            index[i] = state[i] - np.var(state[:-1][self.adjacency_matrix[i]])
        return (index >= self.threshold).astype(int)

    def get_param(self):
        return None


def eval_simple_policy(env, policy, nb_iter=1000, nb_run=1, policy_name=None):
    """
    Evaluate a simple policy

    Parameters
    ----------
    env: gym.Env
        An environment with the gym interface
    policy: SimplePolicy
        A policy with the SimplePolicy interface
    nb_iter: int
        The number of step to evaluate the policy (unless the episode is done before)
    nb_run: int
        The number of run (the policy and the environment are reinitialized at each run)
    policy_name: str
        The name of the policy

    Returns
    -------
    results: pd.DataFrame
        A dataframe with the results of the evaluation
    """
    if policy_name is None:
        policy_name = policy.name
    results = pd.DataFrame(
        columns=[
            "Run",
            "Step",
            "Reward",
            "Cumulative reward",
            "Policy parameter",
            "Policy name",
        ]
    )
    for run in range(nb_run):
        state = env.reset()
        policy.reset()
        done = False
        cum_reward = 0
        for i in range(nb_iter):
            if done:
                break
            action = policy(state)
            state, reward, done, _ = env.step(action)
            cum_reward += reward
            results.loc[len(results)] = [
                run,
                i,
                reward,
                cum_reward,
                policy.get_param(),
                policy_name,
            ]
    return results


def compute_optimal_threshold(env, policy, possible_values, nb_iter=100, nb_run=10):
    """
    Compute the optimal threshold for a policy

    Parameters
    ----------
    env: gym.Env
        A forest environment
    policy: ThresholdPolicy
        A policy with the ThresholdPolicy interface
    possible_values: list
        The list of the values to test for the threshold
    nb_iter: int
        The number of step to evaluate the policy (unless the episode is done before)
    nb_run: int
        The number of run (the policy and the environment are reinitialized at each run)

    Returns
    -------
    optimal_threshold: float
        The optimal threshold among the possible values
    """
    results = pd.DataFrame(columns=["Threshold", "Cumulative reward", "Policy name"])
    for t in tqdm(
        possible_values, desc=f"Computing optimal threshold for {policy.name}"
    ):
        policy.threshold = t
        policy.reset()
        tmp_res = eval_simple_policy(env, policy, nb_iter, nb_run, policy.name)
        results.loc[len(results)] = [
            t,
            tmp_res["Cumulative reward"].mean(),
            policy.name,
        ]
    optimal_threshold = results.loc[results["Cumulative reward"].idxmax()]["Threshold"]
    print("Optimal threshold: {}".format(optimal_threshold))
    return optimal_threshold
