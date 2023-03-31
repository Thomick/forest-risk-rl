import numpy as np


class SimplePolicy:
    def __init__(self, nb_tree):
        self.nb_tree = nb_tree

    def __call__(self, state):
        return np.zeros(self.nb_tree)


class ThresholdPolicy(SimplePolicy):
    def __init__(self, nb_tree, threshold):
        self.nb_tree = nb_tree
        self.threshold = threshold

    def __call__(self, state):
        return (np.array(state)[0 : self.nb_tree] >= self.threshold).astype(int)


class CuttingAgePolicy:
    def __init__(self, nb_tree, cutting_age):
        self.cutting_age = cutting_age
        self.age = np.array([0] * nb_tree)

    def __call__(self, state):
        self.age += 1
        to_cut = self.age >= self.cutting_age
        self.age[to_cut] = 0
        return to_cut.astype(int)
