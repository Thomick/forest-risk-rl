from risk import compute_empirical_cvar, compute_group_risk
from agents import ARRLAgent


class ForestGRAgent(ARRLAgent):
    def __init__(self, env, learner_ctor, learner_args={}, beta=0.1, **kwargs):
        self.beta = beta
        super().__init__(env, learner_ctor, learner_args, **kwargs)

    def update_learner(self, state, action, reward, next_state):
        group_risk = compute_group_risk(
            self.env.index_to_state(next_state),
            self.env.group_risk_weights,
        )
        self.learner.update(state, action, reward - self.beta * group_risk, next_state)


class ForestGROnlyAgent(ARRLAgent):
    def __init__(self, env, learner_ctor, learner_args={}, **kwargs):
        super().__init__(env, learner_ctor, learner_args, **kwargs)

    def update_learner(self, state, action, reward, next_state):
        group_risk = compute_group_risk(
            self.env.index_to_state(next_state),
            self.env.group_risk_weights,
        )
        self.learner.update(state, action, -group_risk, next_state)
