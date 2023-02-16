from rlberry.manager import AgentManager, plot_writer_data, evaluate_agents
from rlberry.envs import gym_make
from rlberry.envs.finite import GridWorld
from rlberry.agents import ucbvi
import numpy as np

import environments.RegisterEnvironments as bW

from agent import ARRLAgent
from learners.discreteMDPs.IRL import IRL

horizon = 20
gamma = 0.99

env_name = bW.registerWorld("random-100")

params = {"learner_ctor": IRL}

eval_kwargs = dict(eval_horizon=horizon, n_simulations=20)


agent_manager = AgentManager(
    ARRLAgent,
    (gym_make, dict(id=env_name)),
    n_fit=1,
    fit_budget=1000,
    init_kwargs=params,
    eval_kwargs=eval_kwargs,
)

agent_manager.fit()

_ = plot_writer_data(
    [agent_manager],
    tag="rewards",
    title="Training Episode Cumulative Rewards",
    preprocess_func=lambda x: np.cumsum(x),
    show=True,
)
