# Preliminary code for discrete model

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from utils import make_grid_matrix

nb_tree = 10
nb_iter = 50
do_cutting = False
p_min = 0.0
p_max = 1.0
alpha = 1
H = 20

s = np.zeros(nb_tree)
states = [s.copy()]

for i in range(nb_iter):
    p_growth = np.zeros(nb_tree)
    for j in range(nb_tree):
        p_growth[j] = np.exp((s[j] - np.mean(s))) / np.exp(alpha)
    print(p_growth)
    p_growth.clip((H - s) / H, p_max, out=p_growth)
    for j in range(nb_tree):
        if np.random.rand() < p_growth[j]:
            s[j] += 1
        if s[j] > H:
            if do_cutting:
                s[j] = 0
            else:
                s[j] = H

    states.append(s.copy())
plt.plot(states)
plt.title(f"{nb_tree} trees with complete graph structure (discrete)")
plt.xlabel("Iteration")
plt.ylabel("Height")
plt.show()

col, row = 5, 5  # For grid structures
adjacency_matrix = make_grid_matrix(row, col)
nb_tree = adjacency_matrix.shape[0]

s = np.zeros(nb_tree)
states = [s.copy()]

for i in range(nb_iter):
    p_growth = np.zeros(nb_tree)
    for j in range(nb_tree):
        nb_neighbors = np.sum(adjacency_matrix[j])
        diff = np.sum(s[adjacency_matrix[j] == 1]) / nb_neighbors - s[j]
        p_growth[j] = np.exp(-diff) / np.exp(alpha)
    print(p_growth)
    p_growth.clip((H - s) / H, p_max, out=p_growth)
    for j in range(nb_tree):
        if np.random.rand() < p_growth[j]:
            s[j] += 1
        if s[j] > H:
            if do_cutting:
                s[j] = 0
            else:
                s[j] = H

    states.append(s.copy())
plt.plot(states)
plt.title(f"{nb_tree} trees with complete graph structure (discrete)")
plt.xlabel("Iteration")
plt.ylabel("Height")

plt.figure()
pos = plt.imshow(
    s.reshape((row, col)),
    interpolation="nearest",
)
plt.colorbar(pos)

plt.show()
