from forest_risk_rl.simple_policies import (
    FireBlockThreshold,
    CuttingAgePolicy,
    ThresholdPolicy,
    ExpertPolicy,
    eval_simple_policy,
    SynchronizedCutting,
    compute_optimal_threshold,
)
from forest_risk_rl.utils import build_transition_matrix, make_grid_matrix
from forest_risk_rl.envs.linear_dynamic_env import ForestWithStorms, ForestWithFires
from forest_risk_rl.risk_measures import compute_empirical_cvar

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


# Parameters for the experiments
row, col = 5, 5  # For grid structures
nb_tree = row * col
alpha = 0.1
beta = 0.05
H = 20  # Asymptotic height parameter for the trees
nb_iter = 100  # Duration of the experiment
nb_rep = 10  # Number of repetitions of the experiment

cvar_level = 0.95
show_plots = False
save_path = "figs"  # Path to save the results, if None, no saving

# Toggle experiments
exp_forest_fire = False
exp_cutting_age = False
exp_storm = True

adjacency_matrix = make_grid_matrix(row, col)
high_fire_risk_env = ForestWithFires(
    nb_tree,
    adjacency_matrix,
    H=H,
    alpha=alpha,
    beta=beta,
    fire_prob=0.2,
    fire_duration=10,
)

low_fire_risk_env = ForestWithFires(
    nb_tree,
    adjacency_matrix,
    H=H,
    alpha=alpha,
    beta=beta,
    fire_prob=0.01,
    fire_duration=10,
)

high_storm_risk_env = ForestWithStorms(
    nb_tree,
    adjacency_matrix,
    H=H,
    alpha=alpha,
    beta=beta,
    storm_prob=0.2,
    storm_power=3,
)

low_storm_risk_env = ForestWithStorms(
    nb_tree,
    adjacency_matrix,
    H=H,
    alpha=alpha,
    beta=beta,
    storm_prob=0.01,
    storm_power=3,
)

fire_env_dict = {
    "High fire risk": high_fire_risk_env,
    "Low fire risk": low_fire_risk_env,
}
storm_env_dict = {
    "High storm risk": high_storm_risk_env,
    "Low storm risk": low_storm_risk_env,
}


def forest_fire_experiment(env, nb_iter, nb_rep, cvar_level, env_key=""):
    results_threshold = eval_simple_policy(
        env, ThresholdPolicy(env.nb_tree, 15), nb_iter, nb_rep, "Threshold policy"
    )
    cvar_threshold = compute_empirical_cvar(
        results_threshold.loc[results_threshold["Step"] == nb_iter - 1][
            "Cumulative reward"
        ],
        cvar_level,
    )
    print(f"Env : {env_key} -> CVaR for Threshold policy: {cvar_threshold}")
    results_fireblock = eval_simple_policy(
        env,
        FireBlockThreshold(env.nb_tree, 15, row, col),
        nb_iter,
        nb_rep,
        "Fireblock policy",
    )
    cvar_fireblock = compute_empirical_cvar(
        results_fireblock.loc[results_threshold["Step"] == nb_iter - 1][
            "Cumulative reward"
        ],
        cvar_level,
    )
    print(f"Env : {env_key} -> CVaR for Fireblock policy: {cvar_fireblock}")

    results = pd.concat([results_threshold, results_fireblock])
    return results


def exp_and_plot(
    env_dict,
    experiment,
    nb_iter,
    nb_rep,
    cvar_level,
    show_plots,
    save_path=None,
    exp_name=None,
):
    if save_path is not None and exp_name is not None:
        save_path = save_path + "/" + exp_name
        import os

        os.makedirs(save_path, exist_ok=True)

    results_list = []
    results_list += Parallel(n_jobs=12)(
        delayed(experiment)(env, nb_iter, nb_rep, cvar_level, key)
        for key, env in env_dict.items()
    )
    results = pd.concat(
        results_list, keys=env_dict.keys(), names=["Environment"]
    ).reset_index()
    # print(results)

    sns.lineplot(
        data=results,
        x="Step",
        y="Cumulative reward",
        hue="Policy name",
        style="Environment",
    )
    if save_path is not None:
        plt.savefig(save_path + "/rewards.png")
    plt.figure()
    sns.boxplot(
        data=results.loc[results["Step"] == nb_iter - 1],
        x="Policy name",
        y="Cumulative reward",
        hue="Environment",
    )
    plt.title("Distribution of the cumulative reward at the end of the experiment")
    if save_path is not None:
        plt.savefig(save_path + "/box.png")
    sns.displot(
        data=results.loc[results["Step"] == nb_iter - 1],
        x="Cumulative reward",
        hue="Policy name",
        col="Environment",
        kind="kde",
    )
    if save_path is not None:
        plt.savefig(save_path + "/dist.png")
    if show_plots:
        plt.show()


if exp_forest_fire:
    print("Forest fire experiment ...")
    exp_and_plot(
        fire_env_dict,
        forest_fire_experiment,
        nb_iter,
        nb_rep,
        cvar_level,
        show_plots,
        save_path=save_path,
        exp_name="forest_fire_experiment",
    )


def cutting_age_experiment(env, nb_iter, nb_rep, cvar_level, env_key=""):
    results_alternated = eval_simple_policy(
        env, CuttingAgePolicy(env.nb_tree, 7), nb_iter, nb_rep, "Cutting age policy"
    )
    cvar = compute_empirical_cvar(
        results_alternated.loc[results_alternated["Step"] == nb_iter - 1][
            "Cumulative reward"
        ],
        cvar_level,
    )
    print(f"Env : {env_key} -> CVaR for Cutting age policy: {cvar}")
    results_synchro = eval_simple_policy(
        env,
        SynchronizedCutting(env.nb_tree, 7),
        nb_iter,
        nb_rep,
        "Synchronized cutting policy",
    )
    cvar = compute_empirical_cvar(
        results_synchro.loc[results_synchro["Step"] == nb_iter - 1][
            "Cumulative reward"
        ],
        cvar_level,
    )
    print(f"Env : {env_key} -> CVaR for Synchronized cutting policy: {cvar}")

    return pd.concat([results_alternated, results_synchro])


if exp_cutting_age:
    print("Cutting age experiment ...")
    exp_and_plot(
        fire_env_dict,
        cutting_age_experiment,
        nb_iter,
        nb_rep,
        cvar_level,
        show_plots,
        save_path=save_path,
        exp_name="cutting_age_experiment",
    )


def storm_experiment(env, nb_iter, nb_rep, cvar_level, env_key=""):
    optimal_threshold = compute_optimal_threshold(
        env, ThresholdPolicy(env.nb_tree, 10), list(range(2, 20)), 100, 20
    )
    results_threshold = eval_simple_policy(
        env,
        ThresholdPolicy(env.nb_tree, optimal_threshold),
        nb_iter,
        nb_rep,
        "Threshold policy",
    )
    cvar = compute_empirical_cvar(
        results_threshold.loc[results_threshold["Step"] == nb_iter - 1][
            "Cumulative reward"
        ],
        cvar_level,
    )
    print(f"Env : {env_key} -> CVaR for Storm policy: {cvar}")

    optimal_threshold = compute_optimal_threshold(
        env,
        ExpertPolicy(env.nb_tree, 10, env.adjacency_matrix),
        list(range(2, 20)),
        100,
        20,
    )
    results_expert = eval_simple_policy(
        env,
        ExpertPolicy(env.nb_tree, 10, env.adjacency_matrix),
        nb_iter,
        nb_rep,
        "Some expert policy",
    )
    cvar = compute_empirical_cvar(
        results_expert.loc[results_expert["Step"] == nb_iter - 1]["Cumulative reward"],
        cvar_level,
    )
    print(f"Env : {env_key} -> CVaR for Some expert policy: {cvar}")

    return pd.concat([results_threshold, results_expert])


if exp_storm:
    print("Storm experiment ...")
    exp_and_plot(
        storm_env_dict,
        storm_experiment,
        nb_iter,
        nb_rep,
        cvar_level,
        show_plots,
        save_path=save_path,
        exp_name="storm_experiment",
    )
