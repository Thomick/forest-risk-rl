# Code for the experiments with a RL agent that learns to harvest trees


from linear_dynamic_env import ForestLinearEnv, ForestWithStorms
from utils import make_grid_matrix, make_octo_grid_matrix

from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


row, col = 5, 5
H = 20
nb_run = 100
nb_iter = 100
nb_epoch = 600
budget = 100
alpha = 0.2
beta = 0.1
with_storms = True
storm_probability = 0.0
load_model = True
nb_model = 4

training_experiment = False
storm_risk_experiment = False
track_risk_experiment = True

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

if training_experiment:
    agent = PPO("MlpPolicy", env, verbose=0)
    if load_model:
        agent = PPO.load("ppo_forest", env)
        nb_epoch = 1
    else:
        agent.learn(total_timesteps=budget)

    df = pd.DataFrame(columns=["Epoch", "Total reward", "Average risk"])

    states = []
    actions = []
    for j in tqdm(range(nb_epoch), desc="Evaluating agent"):
        if not load_model:
            agent.learn(total_timesteps=budget)
        r = []
        states = []
        for k in range(nb_run):
            risks = []
            observation = env.reset()
            states.append(observation)
            total_reward = 0
            for i in range(nb_iter):
                action, _ = agent.predict(observation)
                observation, reward, done, _ = env.step(action)
                total_reward += reward
                states.append(observation)
                actions.append(action)
                risks.append(np.mean(env.compute_risks(observation)[0]))
            df.loc[len(df)] = [j, total_reward, np.mean(risks)]
            r.append(total_reward)
        print(np.mean(r))
    if not load_model:
        agent.save("ppo_forest")
    # print(df)
    sns.lineplot(data=df, x="Epoch", y="Total reward")
    plt.title("Total reward of learned policy during training")
    plt.figure()
    sns.lineplot(data=df, x="Epoch", y="Average risk")
    plt.title("Average risk of learned policy during training")
    plt.figure()
    plt.plot(states[0:nb_iter])
    action_array = np.zeros(nb_tree)
    for a in actions:
        action_array += a
    plt.figure()
    plt.imshow(action_array.reshape(row, col) / nb_run)
    plt.colorbar()
    plt.show()


def run_experiment(storm_prob, model_id):
    print(f"Starting training of model {model_id} for storm probability {storm_prob}")
    env = ForestWithStorms(
        nb_tree,
        adjacency_matrix,
        H=H,
        alpha=alpha,
        beta=beta,
        storm_prob=storm_prob,
    )
    if load_model:
        agent = PPO.load(
            f"checkpoints/ppo_forest_storm_prob_{storm_prob}_model_{model_id}", env
        )
        nb_epoch = 0
    else:
        agent = PPO("MlpPolicy", env, verbose=0)

    run_df = pd.DataFrame(columns=["Storm probability", "Total reward", "Average risk"])

    for _ in range(nb_epoch):
        agent.learn(
            total_timesteps=budget,
            tb_log_name=f"runs/storm_prob_{storm_prob}_model_{model_id}",
        )

    states = []
    actions = []
    for k in range(nb_run):
        risks = []
        observation = env.reset()
        states.append(observation)
        total_reward = 0
        for i in range(nb_iter):
            action, _ = agent.predict(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            states.append(observation)
            actions.append(action)
            risks.append(np.mean(env.compute_risks(observation)[0]))
        run_df.loc[len(run_df)] = [storm_prob, total_reward, np.mean(risks)]

    action_array = np.zeros(nb_tree)
    for a in actions:
        action_array += a
    plt.imshow(action_array.reshape(row, col))
    plt.colorbar()
    plt.title(f"Average number of harvests for storm probability {storm_prob}")
    plt.savefig(f"plots/2_storm_prob_{storm_prob}_model_{model_id}.png")
    plt.close()
    if not load_model:
        agent.save(f"checkpoints/ppo_forest_storm_prob_{storm_prob}_model_{model_id}")
    print(f"Finished training of model {model_id} for storm probability {storm_prob}")

    return run_df


if storm_risk_experiment:
    df_list = []
    for storm_prob in tqdm([0.0, 0.05, 0.1, 0.2, 0.35, 0.5]):
        df_list += Parallel(n_jobs=4)(
            delayed(run_experiment)(storm_prob, i) for i in range(nb_model)
        )
    exp_df = pd.concat(df_list)
    exp_df.to_csv("storm_risk_experiment.csv")
    sns.lineplot(data=exp_df, x="Storm probability", y="Total reward")
    plt.title("Total reward of learned policy")
    plt.savefig("storm_risk_experiment_reward.png")
    plt.figure()
    sns.lineplot(data=exp_df, x="Storm probability", y="Average risk")
    plt.title("Average risk of learned policy")
    plt.savefig("storm_risk_experiment_risk.png")
    plt.show()

if track_risk_experiment:
    agent = PPO.load("ppo_forest", env)
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
            action, _ = agent.predict(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            df.loc[len(df)] = [
                i,
                reward,
                env.compute_risks(observation)[0].mean(),
                observation[0],
            ]
    sns.lineplot(data=df, x="Step", y="Reward")
    plt.title(f"Reward over time (PPO learned policy)")
    plt.figure()
    sns.lineplot(data=df, x="Step", y="Windthrow risk")
    plt.title(f"Windthrow risk over time (PPO learned policy)")
    plt.figure()
    sns.lineplot(data=df, x="Step", y="Height_0")
    plt.show()
