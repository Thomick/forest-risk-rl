# Code for the experiments with a simple cutting age based policy

import numpy as np

from utils import build_transition_matrix, make_grid_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from linear_dynamic_env import ForestLinearEnv, ForestWithStorms


row, col = 5, 5
nb_iter = 100
nb_run = 100
period = 20
H = 20
sync = False
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
    env = ForestLinearEnv(nb_tree, adjacency_matrix, H=H, alpha=alpha, beta=alpha)


all_total_rewards = []
for _ in range(nb_run):
    env.reset()
    states = []
    total_reward = 0
    for i in range(nb_iter - 1):
        action = [0] * nb_tree
        for j in range(nb_tree):
            action[j] = 1 if (i + j) % period == 0 else 0
        observation, reward, done, _ = env.step(action)
        states.append(observation)
        total_reward += reward
    if nb_run == 1:
        plt.plot(states)
        plt.show()
    all_total_rewards.append(total_reward)
sns.boxplot(all_total_rewards)
print(np.mean(all_total_rewards))
plt.show()

df = pd.DataFrame(
    columns=[
        "Cutting age",
        "Total reward",
        "Number of harvest",
        "Average harvest size",
    ]
)
for p in tqdm(range(1, 31), desc="Computing optimal period"):
    observation = env.reset()
    for _ in range(nb_run):
        harvest_sizes = []
        env.reset()
        total_reward = 0
        harvest_count = 0
        for i in range(nb_iter - 1):
            action = [0] * nb_tree
            for j in range(nb_tree):
                if ((i + j) % p == 0 and not sync) or ((i) % p == 0 and sync):
                    action[j] = 1
                    harvest_sizes.append(observation[j])
                    harvest_count += 1
            observation, reward, done, _ = env.step(action)
            total_reward += reward
        df.loc[len(df)] = [p, total_reward, harvest_count, np.mean(harvest_sizes)]


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.lineplot(
    data=df,
    x="Cutting age",
    y="Total reward",
    color="blue",
    ax=ax1,
)
sns.lineplot(
    data=df,
    x="Cutting age",
    y="Average harvest size",
    ax=ax2,
    color="red",
)
ax1.set_ylabel("Total reward", color="blue")
ax2.set_ylabel("Average harvest size", color="red")
plt.tight_layout()
plt.show()
