# Simple and expert policies

import numpy as np


class SimplePolicy:
    """
    A simple policy that always never cut any tree
    """

    def __init__(self, nb_tree):
        self.nb_tree = nb_tree

    def __call__(self, state):
        return np.zeros(self.nb_tree)


class ThresholdPolicy(SimplePolicy):
    """
    A policy that always cut the tree if its height is above a threshold
    """

    def __init__(self, nb_tree, threshold):
        self.nb_tree = nb_tree
        self.threshold = threshold

    def __call__(self, state):
        return (np.array(state)[0 : self.nb_tree] >= self.threshold).astype(int)


class CuttingAgePolicy(SimplePolicy):
    """
    A policy that always cut the tree if its age is above a threshold
    """

    def __init__(self, nb_tree, cutting_age):
        self.cutting_age = cutting_age
        self.age = np.array([0] * nb_tree)

    def __call__(self, state):
        self.age += 1
        to_cut = self.age >= self.cutting_age
        self.age[to_cut] = 0
        return to_cut.astype(int)


class FireBlockThreshold(ThresholdPolicy):
    """
    A policy that repeatedly cut the trees such that the fire does not spread to the entire forest.
    The forest is partitionned in 4 quadrants separated by areas with not trees.
    Inside each quadrant, the trees are cut if their height is above a threshold.
    """

    def __init__(self, nb_tree, threshold, n_row, n_col):
        super().__init__(nb_tree, threshold)
        self.n_row = n_row
        self.n_col = n_col
        self.blocker_mask = np.zeros(
            (n_row, n_col)
        )  # The fire is restrained by partitionning the in smaller grids separated by area with not trees
        for i in range(n_row):
            for j in range(n_col):
                if i == n_row // 2 or j == n_col // 2:
                    self.blocker_mask[i, j] = 1
        print(self.blocker_mask)

    def __call__(self, state):
        to_cut = super().__call__(state)
        return to_cut * self.blocker_mask.flatten()
