import numpy as np

from utils import build_transition_matrix, make_grid_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from models.linear_dynamic_env import ForestLinearEnv, ForestWithStorms


row, col = 5, 5
nb_iter = 100
nb_run = 100
H = 20
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


df = pd.DataFrame(
    columns=[
        "Threshold",
        "Total reward",
        "Number of harvest",
        "Average harvest size",
        "Average cutting age",
    ]
)
for t in tqdm(range(1, 23), desc="Computing optimal threshold"):
    observation = env.reset()
    for _ in range(nb_run):
        harvest_sizes = []
        env.reset()
        total_reward = 0
        harvest_count = 0
        time_last_harvest = [0] * nb_tree
        cutting_age = []
        for i in range(nb_iter - 1):
            action = [0] * nb_tree
            for j in range(nb_tree):
                if observation[j] >= t:
                    action[j] = 1
                    harvest_sizes.append(observation[j])
                    harvest_count += 1
                    cutting_age.append(i - time_last_harvest[j])
                    time_last_harvest[j] = i
            observation, reward, done, _ = env.step(action)
            total_reward += reward
        df.loc[len(df)] = [
            t,
            total_reward,
            harvest_count,
            np.mean(harvest_sizes),
            np.mean(cutting_age),
        ]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.lineplot(
    data=df,
    x="Threshold",
    y="Total reward",
    color="blue",
    ax=ax1,
)
sns.lineplot(
    data=df,
    x="Threshold",
    y="Average cutting age",
    ax=ax2,
    color="red",
)
ax1.set_ylabel("Total reward", color="blue")
ax2.set_ylabel("Average cutting age", color="red")
plt.tight_layout()
plt.show()
