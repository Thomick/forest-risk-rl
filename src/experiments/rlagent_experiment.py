from rlberry.manager import (
    AgentManager,
    plot_writer_data,
    read_writer_data,
    evaluate_agents,
    MultipleManagers,
)
from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import PPO

from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models.linear_dynamic_env import ForestLinearEnv, ForestWithStorms
from utils import build_transition_matrix, make_grid_matrix


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


row, col = 5, 5
H = 20
nb_run = 500
nb_iter = 100
budget = 100
alpha = 0.2
beta = 0.1
with_storms = True
storm_probability = 0.05

nb_tree = row * col
adjacency_matrix = make_grid_matrix(row, col)
if with_storms:
    env = ForestWithStorms(
        nb_tree,
        adjacency_matrix,
        H=H,
        alpha=alpha,
        beta=beta,
        storm_prob=storm_probability,
    )
else:
    env = ForestLinearEnv(nb_tree, adjacency_matrix, H=H, alpha=alpha, beta=beta)

agent = PPO("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=budget)

df = pd.DataFrame(columns=["Run", "Total reward"])

states = []
actions = []
for j in tqdm(range(nb_run), desc="Evaluating agent"):
    agent.learn(total_timesteps=budget)
    all_total_rewards = []
    harvest_sizes = []
    observation = env.reset()
    if j == (nb_run - 1):
        states.append(observation)
    total_reward = 0
    harvest_count = 0
    time_last_harvest = [0] * nb_tree
    cutting_age = []
    for i in range(nb_iter):
        action, _ = agent.predict(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if j == (nb_run - 1):
            states.append(observation)
            actions.append(action)
    df.loc[len(df)] = [
        j,
        total_reward,
    ]
print(df)
sns.lineplot(data=df, x="Run", y="Total reward")
plt.figure()
plt.plot(states)
action_array = np.zeros(nb_tree)
for a in actions:
    action_array += a
plt.figure()
plt.imshow(action_array.reshape(row, col))
plt.colorbar()
plt.show()
