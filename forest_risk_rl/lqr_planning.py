from forest_risk_rl.utils import (
    build_transition_matrix,
    make_grid_matrix,
    make_octo_grid_matrix,
)

from forest_risk_rl.envs.linear_dynamic_env import ForestLinearEnv
from forest_risk_rl.simple_policies import ThresholdPolicy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Parameters
row, col = 5, 5  # For grid structures
nb_tree = row * col
alpha = 0.2
beta = 0.1
H = 20  # Asymptotic height parameter for the trees
nb_steps = 1000  # Duration of the experiment


env = ForestLinearEnv(nb_tree, make_grid_matrix(row, col), H, alpha, beta)
A = env.A
B = env.B
R = -1000 * env.N
Q = np.eye(nb_tree + 1)

K = np.zeros((nb_steps, nb_tree + 1, nb_tree + 1))
L = np.zeros((nb_steps, nb_tree, nb_tree + 1))
for t in range(nb_steps).__reversed__():
    if t < nb_steps - 1:
        K[t] = (
            A.T
            @ (
                K[t + 1]
                - K[t + 1] @ B @ np.linalg.inv(B.T @ K[t + 1] @ B + R) @ B.T @ K[t + 1]
                + Q
            )
            @ A
        )
        L[t] = -np.linalg.inv(B.T @ K[t + 1] @ B + R) @ B.T @ K[t + 1] @ A
    elif t == nb_steps - 1:
        K[t] = Q
observation = env.reset()


states = [observation]
rewards = []
true_actions = []
for cur_t in tqdm(range(nb_steps)):
    """K = np.zeros((nb_steps - cur_t, nb_tree + 1, nb_tree + 1))
    L = np.zeros((nb_steps - cur_t, nb_tree, nb_tree + 1))
    for t in range(0, nb_steps - cur_t).__reversed__():
        if t < nb_steps - cur_t - 1:
            K[t] = (
                A.T
                @ (
                    K[t + 1]
                    - K[t + 1]
                    @ B
                    @ np.linalg.inv(B.T @ K[t + 1] @ B + R)
                    @ B.T
                    @ K[t + 1]
                    + Q
                )
                @ A
            )
            L[t] = -np.linalg.inv(B.T @ K[t + 1] @ B + R) @ B.T @ K[t + 1] @ A
        elif t == nb_steps - cur_t - 1:
            K[t] = Q"""
    action = L[cur_t] @ observation
    true_action = action <= -observation[:-1] / 2
    true_actions.append(true_action)
    observation, reward, done, info = env.step(true_action)
    states.append(observation)
    rewards.append(reward)

plt.plot(states)
plt.figure()
plt.plot(np.cumsum(rewards))
plt.figure()
plt.plot(np.linalg.norm(L, axis=(1, 2)))
print(np.sum(rewards))
plt.figure()
plt.imshow(np.sum(true_actions, axis=0).reshape((row, col)))
plt.colorbar()
plt.figure()

observation = env.reset()
threshold_policy = ThresholdPolicy(nb_tree, 15)

states = [observation]
rewards = []
true_actions = []
L_memory = []
for cur_t in tqdm(range(nb_steps)):
    true_action = threshold_policy(observation)
    true_actions.append(true_action)
    observation, reward, done, info = env.step(true_action)
    states.append(observation)
    rewards.append(reward)
    L_memory.append(np.linalg.norm(L[0]))

plt.plot(states)
plt.figure()
plt.plot(np.cumsum(rewards))
print(np.sum(rewards))
plt.figure()
plt.imshow(np.sum(true_actions, axis=0).reshape((row, col)))
plt.colorbar()
plt.show()

observation = env.reset()
