from rlberry.manager import (
    AgentManager,
    plot_writer_data,
    read_writer_data,
    evaluate_agents,
    MultipleManagers,
)
from rlberry.envs import gym_make
import numpy as np

import environments.RegisterEnvironments as bW
from env import register_forestmdp

from agents import ARRLAgent, OptAgent
from learners.discreteMDPs.IRL import IRL
from learners.Generic.Qlearning import Qlearning
from learners.discreteMDPs.PSRL import PSRL

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_opti_average_reward(env_name, budget):
    agent_manager = AgentManager(
        OptAgent,
        (gym_make, dict(id=env_name)),
        n_fit=1,
        fit_budget=budget,
    )

    agent_manager.fit()

    opti_writer_df = agent_manager.get_writer_data()[0]
    opti_average_rewards = opti_writer_df[opti_writer_df["tag"] == "rewards"][
        "value"
    ].values
    # plt.plot(opti_average_rewards.cumsum())
    return opti_average_rewards.mean()


gamma = 1.0
n_fit = 3
budget = 10000
plot_regret = True
plot_group_risk = True

# env_name = bW.registerWorld("random-100")
env_name = register_forestmdp(
    nbGrowState=10, nbNeighbors=3, Pg=0.95, Pw=0.05, model_type="independent"
)

if __name__ == "__main__":
    eval_kwargs = dict(eval_horizon=100, n_simulations=20, metric="cvar")

    multi_manager = MultipleManagers(parallelization="process")
    """
    multi_manager.append(
        AgentManager(
            ARRLAgent,
            (gym_make, dict(id=env_name)),
            n_fit=n_fit,
            fit_budget=budget,
            init_kwargs={"learner_ctor": IRL},
            eval_kwargs=eval_kwargs,
            agent_name="IRL",
        )
    )
    multi_manager.append(
        AgentManager(
            ARRLAgent,
            (gym_make, dict(id=env_name)),
            n_fit=n_fit,
            fit_budget=budget,
            init_kwargs={"learner_ctor": PSRL, "learner_args": {"delta": 0.05}},
            eval_kwargs=eval_kwargs,
            agent_name="PSRL",
        )
    )"""
    multi_manager.append(
        AgentManager(
            ARRLAgent,
            (gym_make, dict(id=env_name)),
            n_fit=n_fit,
            fit_budget=budget,
            init_kwargs={
                "learner_ctor": Qlearning,
                "render": False,
                "track_group_risk": plot_group_risk,
            },
            eval_kwargs=eval_kwargs,
            agent_name="Qlearning",
        )
    )

    multi_manager.run()

    if plot_regret:
        opti_average_reward = get_opti_average_reward(env_name, budget)
        print(opti_average_reward)

    _ = plot_writer_data(
        multi_manager.managers,
        tag="rewards",
        title="Training Regret" if plot_regret else "Training Cumulated Reward",
        preprocess_func=lambda x: np.cumsum(opti_average_reward - x)
        if plot_regret
        else np.cumsum(x),
        show=True,
    )

    if plot_group_risk:
        _ = plot_writer_data(
            multi_manager.managers,
            tag="group_risk",
            title="Group Risk",
            preprocess_func=lambda x: x,
            show=True,
        )

    evaluate_agents(multi_manager.managers, plot=True, show=True)
