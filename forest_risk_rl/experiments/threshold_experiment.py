# Code for the experiments with a simple threshold policy

import numpy as np

from forest_risk_rl.utils import make_grid_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from forest_risk_rl.envs.linear_dynamic_env import ForestLinearEnv, ForestWithStorms


row, col = 5, 5
nb_iter = 100
nb_run = 100
H = 20
alpha = 0.2
beta = 0.1
with_storms = True
storm_probability = 0.05
default_threshold = 15

optimal_threshold_experiment = True
storm_impact_experiment = False
track_risk_experiment = False


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

if optimal_threshold_experiment:
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

if storm_impact_experiment:
    df = pd.DataFrame(
        columns=[
            "Threshold",
            "Total reward",
            "Number of harvest",
            "Average harvest size",
            "Average cutting age",
            "Storm probability",
            "Storm power",
            "Average windthrow risk",
        ]
    )
    for prob in tqdm(
        [0.0, 0.05, 0.1, 0.2, 0.35, 0.5], desc="Testing multiple storm probabilities"
    ):
        env.storm_prob = prob
        for power in [1, 2, 3, 4, 5]:
            env.D = power
            observation = env.reset()
            for _ in range(nb_run):
                harvest_sizes = []
                env.reset()
                total_reward = 0
                harvest_count = 0
                time_last_harvest = [0] * nb_tree
                cutting_age = []
                risks = []
                for i in range(nb_iter - 1):
                    action = [0] * nb_tree
                    for j in range(nb_tree):
                        if observation[j] >= default_threshold:
                            action[j] = 1
                            harvest_sizes.append(observation[j])
                            harvest_count += 1
                            cutting_age.append(i - time_last_harvest[j])
                            time_last_harvest[j] = i
                    observation, reward, done, _ = env.step(action)
                    total_reward += reward
                    risks.append(np.mean(env.compute_risks(observation)[0]))
                df.loc[len(df)] = [
                    default_threshold,
                    total_reward,
                    harvest_count,
                    np.mean(harvest_sizes),
                    np.mean(cutting_age),
                    prob,
                    power,
                    np.mean(risks),
                ]

    sns.heatmap(
        df.groupby(["Storm probability", "Storm power"])
        .mean()["Total reward"]
        .unstack(),
        annot=True,
        fmt=".1f",
    )
    plt.title(
        f"Total reward with storm risks (threshold = {default_threshold}, over {nb_iter} steps)"
    )
    plt.figure()
    sns.lineplot(
        data=df,
        x="Storm probability",
        y="Average windthrow risk",
        hue="Storm power",
    )
    plt.figure()
    sns.lineplot(
        data=df,
        x="Storm probability",
        y="Total reward",
        hue="Storm power",
    )
    # plt.tight_layout()
    plt.show()

if track_risk_experiment:
    df = pd.DataFrame(
        columns=[
            "Step",
            "Reward",
            "Windthrow risk",
            "Height_0",
        ]
    )
    for _ in range(nb_run):
        harvest_sizes = []
        observation = env.reset()
        total_reward = 0
        harvest_count = 0
        for i in range(nb_iter - 1):
            action = [0] * nb_tree
            for j in range(nb_tree):
                for j in range(nb_tree):
                    if observation[j] >= default_threshold:
                        action[j] = 1
                        harvest_sizes.append(observation[j])
                        harvest_count += 1
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            df.loc[len(df)] = [
                i,
                reward,
                env.compute_risks(observation)[0].mean(),
                observation[0],
            ]
    sns.lineplot(data=df, x="Step", y="Reward")
    plt.title(f"Reward over time (threshold = {default_threshold})")
    print(df)
    plt.figure()
    sns.lineplot(data=df, x="Step", y="Windthrow risk")
    plt.title(f"Windthrow risk over time (threshold = {default_threshold})")
    plt.figure()
    sns.lineplot(data=df, x="Step", y="Height_0")
    plt.show()
