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
default_cutting_age = 7
with_storms = True
storm_probability = 0.2

dispersion_experiment = False
optimal_age_experiment = False
storm_impact_experiment = True
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
    env = ForestLinearEnv(nb_tree, adjacency_matrix, H=H, alpha=alpha, beta=alpha)

if dispersion_experiment:
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

if optimal_age_experiment:
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

if storm_impact_experiment:
    df = pd.DataFrame(
        columns=[
            "Cutting age",
            "Total reward",
            "Number of harvest",
            "Average harvest size",
            "Storm probability",
            "Storm power",
            "Average windthrow risk",
        ]
    )
    for prob in tqdm(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], desc="Testing multiple storm probabilities"
    ):
        env.storm_prob = prob
        for power in [1, 2, 3, 4, 5]:
            env.D = power
            observation = env.reset()
            for _ in range(nb_run):
                harvest_sizes = []
                risks = []
                env.reset()
                total_reward = 0
                harvest_count = 0
                for i in range(nb_iter - 1):
                    action = [0] * nb_tree
                    for j in range(nb_tree):
                        if ((i + j) % default_cutting_age == 0 and not sync) or (
                            (i) % default_cutting_age == 0 and sync
                        ):
                            action[j] = 1
                            harvest_sizes.append(observation[j])
                            harvest_count += 1
                    observation, reward, done, _ = env.step(action)
                    total_reward += reward
                    risks.append(np.mean(env.compute_risks(observation)))
                df.loc[len(df)] = [
                    default_cutting_age,
                    total_reward,
                    harvest_count,
                    np.mean(harvest_sizes),
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
        f"Total reward with storm risks (cutting age = {default_cutting_age}, over {nb_iter} steps)"
    )
    plt.figure()
    sns.lineplot(
        data=df,
        x="Storm probability",
        y="Average windthrow risk",
        hue="Storm power",
    )
    plt.title(
        f"Average windthrow risk of the fixed cutting age policy (cutting age = {default_cutting_age})"
    )
    plt.figure()
    sns.lineplot(
        data=df,
        x="Storm probability",
        y="Total reward",
        hue="Storm power",
    )
    plt.title(
        f"Total reward of the fixed cutting age policy (cutting age = {default_cutting_age})"
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
                if ((i + j) % default_cutting_age == 0 and not sync) or (
                    (i) % default_cutting_age == 0 and sync
                ):
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
    plt.title(f"Reward over time (cutting age = {default_cutting_age})")
    plt.figure()
    sns.lineplot(data=df, x="Step", y="Windthrow risk")
    plt.title(f"Windthrow risk over time (cutting age = {default_cutting_age})")
    plt.figure()
    sns.lineplot(data=df, x="Step", y="Height_0")
    plt.show()
