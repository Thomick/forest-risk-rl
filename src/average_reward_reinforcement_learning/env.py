import gym
from gym.envs.registration import register
import numpy as np
from environments.discreteMDPs.gymWrapper import DiscreteMDP, Dirac
from tqdm import tqdm


class ForestMDP(DiscreteMDP):
    def __init__(self, nbGrowState, nbNeighbors, Pg, Pw, name="ForestMDP"):
        self.group_risk_weights = [
            2 * (nbGrowState - 1 - i) / (nbGrowState * (nbGrowState - 1))
            for i in range(nbGrowState)
        ]

        self.nbNeighbors = nbNeighbors
        self.nbGrowState = nbGrowState
        self.Pg = Pg
        self.Pw = Pw
        self.nS = nbGrowState ** (nbNeighbors + 1)
        self.nA = 2
        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)

        self.startdistribution = np.zeros((self.nS))
        self.rewards = {}
        self.P = {}

        self.init_P_and_rewards()

        self.startdistribution = np.zeros((self.nS))
        start_state = [i % nbGrowState for i in range(nbNeighbors + 1)]
        self.startdistribution[self.state_to_index(start_state)] = 1
        super().__init__(
            self.nS,
            self.nA,
            self.P,
            self.rewards,
            self.startdistribution,
            seed=None,
            name=name,
        )

    def init_P_and_rewards(self):
        for i in tqdm(self.states, desc="Initializing ForestMDP"):
            self.P[i] = {}
            self.rewards[i] = {}
            for a in self.actions:
                self.P[i][a] = []
                if a == 0:
                    self.rewards[i][a] = Dirac(0)
                    cur_state = self.index_to_state(i)
                    new_state = cur_state.copy()
                    for j in range(1, len(new_state)):
                        new_state[j] += 1
                        if new_state[j] >= self.nbGrowState:
                            new_state[j] = 0

                    self.P[i][a].append(
                        (
                            (1 - self.Pw) * (1 - self.Pg),
                            self.state_to_index(new_state),
                            False,
                        )
                    )
                    new_state[0] = min(new_state[0] + 1, self.nbGrowState - 1)
                    self.P[i][a].append(
                        ((1 - self.Pw) * self.Pg, self.state_to_index(new_state), False)
                    )
                    new_state[0] = 0
                    self.P[i][a].append(
                        (self.Pw, self.state_to_index(new_state), False)
                    )
                elif a == 1:
                    cur_state = self.index_to_state(i)
                    self.rewards[i][a] = Dirac(cur_state[0])
                    new_state = cur_state.copy()
                    for j in range(1, len(new_state)):
                        new_state[j] += 1
                        if new_state[j] >= self.nbGrowState:
                            new_state[j] = 0
                    new_state[0] = 0
                    self.P[i][a].append((1, self.state_to_index(new_state), False))

    def state_to_index(self, state):
        index = state[self.nbNeighbors]
        for i in reversed(range(self.nbNeighbors)):
            index = state[i] + index * self.nbGrowState
        return index

    def index_to_state(self, index):
        state = []
        for i in range(self.nbNeighbors + 1):
            state.append(index % self.nbGrowState)
            index = index // self.nbGrowState
        return state


class LocalForestMDP(ForestMDP):
    def __init__(self, nbGrowState, nbNeighbors, Pg, Pw, name="GroupRiskForestMDP"):
        super().__init__(nbGrowState, nbNeighbors, Pg, Pw, name)

    def init_P_and_rewards(self):
        for i in self.states:
            self.P[i] = {}
            self.rewards[i] = {}
            for a in self.actions:
                self.P[i][a] = []
                if a == 0:
                    self.rewards[i][a] = Dirac(0)
                    cur_state = self.index_to_state(i)
                    actual_pW = self.Pw * (
                        1
                        - sum(cur_state[1:])
                        / (self.nbNeighbors * (self.nbGrowState - 1))
                    )
                    new_state = cur_state.copy()
                    for j in range(1, len(new_state)):
                        new_state[j] += 1
                        if new_state[j] >= self.nbGrowState:
                            new_state[j] = 0

                    self.P[i][a].append(
                        (
                            (1 - actual_pW) * (1 - self.Pg),
                            self.state_to_index(new_state),
                            False,
                        )
                    )
                    new_state[0] = min(new_state[0] + 1, self.nbGrowState - 1)
                    self.P[i][a].append(
                        (
                            (1 - actual_pW) * self.Pg,
                            self.state_to_index(new_state),
                            False,
                        )
                    )
                    new_state[0] = 0
                    self.P[i][a].append(
                        (actual_pW, self.state_to_index(new_state), False)
                    )
                elif a == 1:
                    cur_state = self.index_to_state(i)
                    self.rewards[i][a] = Dirac(cur_state[0])
                    new_state = cur_state.copy()
                    for j in range(1, len(new_state)):
                        new_state[j] += 1
                        if new_state[j] >= self.nbGrowState:
                            new_state[j] = 0
                    new_state[0] = 0
                    self.P[i][a].append((1, self.state_to_index(new_state), False))


def register_forestmdp(
    nbGrowState, nbNeighbors, Pg=0.5, Pw=0.1, model_type="independent"
):
    name = (
        "ForestMDP-S" + str(nbGrowState) + "N" + str(nbNeighbors) + model_type + "-v0"
    )
    if model_type == "independent":
        register(
            id=name,
            entry_point="env:ForestMDP",
            max_episode_steps=np.infty,
            reward_threshold=np.infty,
            kwargs={
                "nbGrowState": nbGrowState,
                "nbNeighbors": nbNeighbors,
                "Pg": Pg,
                "Pw": Pw,
                "name": name,
            },
        )
    elif model_type == "local":
        register(
            id=name,
            entry_point="env:LocalForestMDP",
            max_episode_steps=np.infty,
            reward_threshold=np.infty,
            kwargs={
                "nbGrowState": nbGrowState,
                "nbNeighbors": nbNeighbors,
                "Pg": Pg,
                "Pw": Pw,
                "name": name,
            },
        )
    else:
        raise ValueError("model_type must be either independent or local")
    return name
