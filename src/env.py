import gym
from gym.envs.registration import register
import numpy as np
from environments.discreteMDPs.gymWrapper import DiscreteMDP, Dirac


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
        for i in self.states:
            self.P[i] = {}
            self.rewards[i] = {}
            for a in self.actions:
                self.P[i][a] = []
                cur_state = self.index_to_state(i)
                actual_pW = self.computePw(cur_state)
                if a == 0:
                    possible_states = self.list_possible_states(cur_state, 0, actual_pW)
                    self.rewards[i][a] = Dirac(0)
                    self.P[i][a] = [
                        (ps[0], self.state_to_index(ps[1]), ps[2])
                        for ps in possible_states
                    ]
                elif a == 1:
                    possible_states = self.list_possible_states(cur_state, 1, actual_pW)
                    self.rewards[i][a] = Dirac(cur_state[0])
                    self.P[i][a] = [
                        (ps[0], self.state_to_index([0] + ps[1]), ps[2])
                        for ps in possible_states
                    ]

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

    def list_possible_states(self, current_state, current_plot, actual_Pw):
        if current_plot == self.nbNeighbors + 1:
            return [(1, [], False)]
        possible_states = self.list_possible_states(
            current_state, current_plot + 1, actual_Pw
        )
        growth_transition = [
            (
                self.Pg * (1 - actual_Pw) * partial_state[0],
                [min(current_state[current_plot] + 1, self.nbGrowState - 1)]
                + partial_state[1],
                False,
            )
            for partial_state in possible_states
        ]
        stay_transition = [
            (
                (1 - self.Pg) * (1 - actual_Pw) * partial_state[0],
                [current_state[current_plot]] + partial_state[1],
                False,
            )
            for partial_state in possible_states
        ]
        windthrow_transition = [
            (
                actual_Pw * partial_state[0],
                [0] + partial_state[1],
                False,
            )
            for partial_state in possible_states
        ]

        return growth_transition + stay_transition + windthrow_transition

    def computePw(self, state):
        return self.Pw


class LocalForestMDP(ForestMDP):
    def __init__(self, nbGrowState, nbNeighbors, Pg, Pw, name="LocalForestMDP"):
        super().__init__(nbGrowState, nbNeighbors, Pg, Pw, name)

    def computePw(self, state):
        return self.Pw * (
            1 - sum(state) / ((self.nbNeighbors + 1) * (self.nbGrowState - 1))
        )


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
