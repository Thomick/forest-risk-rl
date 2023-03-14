# Code for the experiments with a RL agent that learns to harvest trees

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

from linear_dynamic_env import ForestLinearEnv, ForestWithStorms
from utils import build_transition_matrix, make_grid_matrix, make_octo_grid_matrix


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


row, col = 5, 5
H = 20
nb_run = 10
nb_iter = 100
nb_epoch = 500
budget = 100
alpha = 0.2
beta = 0.1
with_storms = False
storm_probability = 0.05
load_model = False

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
if load_model:
    agent = PPO.load("ppo_forest", env)
    nb_epoch = 1
else:
    agent.learn(total_timesteps=budget)

df = pd.DataFrame(columns=["Epoch", "Total reward"])

states = []
actions = []
for j in tqdm(range(nb_epoch), desc="Evaluating agent"):
    if not load_model:
        agent.learn(total_timesteps=budget)
    r = []
    states = []
    actions = []
    for k in range(nb_run):
        observation = env.reset()
        states.append(observation)
        total_reward = 0
        for i in range(nb_iter):
            action, _ = agent.predict(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            states.append(observation)
            actions.append(action)
        df.loc[len(df)] = [j, total_reward]
        r.append(total_reward)
    print(np.mean(r))
if not load_model:
    agent.save("ppo_forest")
# print(df)
sns.lineplot(data=df, x="Epoch", y="Total reward")
plt.figure()
plt.plot(states[0:nb_iter])
action_array = np.zeros(nb_tree)
for a in actions:
    action_array += a
plt.figure()
plt.imshow(action_array.reshape(row, col) / nb_run)
plt.colorbar()
plt.show()
