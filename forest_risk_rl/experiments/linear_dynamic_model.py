# Preliminary experiments with a linear dynamic model

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Parameters for the experiments
nb_tree = 10
col, row = 5, 5  # For grid structures
alpha = 0.1
beta = 0.05
H = 20  # Asymptotic height parameter for the trees
nb_iter = 1000  # Duration of the experiment
edge_probability = 0.2  # For random Erdos-Renyi graphes
cutting_probability = 0.1
cutting_threshold = H + 1
nb_samples = 1000  # grid experiment

# Toggle experiments
exp_complete_graph = True
exp_random_graph = True
exp_grid = True
grid_plot_asymptotic = True
exp_octo_grid = True
exp_random_actions = True
exp_cutting_threshold = True


def graph_from_adjacency_matrix(adjacency_matrix):
    G = nx.DiGraph()
    for i in range(nb_tree):
        G.add_node(i)
        for j in range(nb_tree):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)
    return G


if exp_complete_graph:
    s = np.random.uniform(0, H, nb_tree + 1)
    s[nb_tree] = H
    states = [s]

    transition_matrix = np.zeros((nb_tree + 1, nb_tree + 1))
    for i in range(nb_tree):
        for j in range(nb_tree):
            transition_matrix[i, j] = -beta / (nb_tree - 1)
        transition_matrix[i, i] = 1 - alpha + beta
        transition_matrix[i, nb_tree] = alpha
    transition_matrix[nb_tree, nb_tree] = 1

    for i in range(nb_iter):
        s = transition_matrix @ s
        states.append(s)

    plt.plot(states)
    plt.title(f"{nb_tree} trees with complete graph structure")
    plt.xlabel("Iteration")
    plt.ylabel("Height")
plt.show()

#######################
# With graph structure


def build_transition_matrix(adjacency_matrix, alpha, beta):
    nb_tree = adjacency_matrix.shape[0]
    transition_matrix = np.zeros((nb_tree + 1, nb_tree + 1))
    for i in range(nb_tree):
        nb_neighbors = np.sum(adjacency_matrix[i, :]) - adjacency_matrix[i, i]
        for j in range(nb_tree):
            if adjacency_matrix[i, j] == 1:
                transition_matrix[i, j] = -beta / nb_neighbors
        transition_matrix[i, i] = 1 - alpha
        if nb_neighbors > 0:
            transition_matrix[i, i] += beta
        transition_matrix[i, nb_tree] = alpha
    transition_matrix[nb_tree, nb_tree] = 1
    return transition_matrix


if exp_random_graph:
    adjacency_matrix = np.random.binomial(1, edge_probability, (nb_tree, nb_tree))
    for i in range(nb_tree):  # Make it symmetric
        for j in range(i, nb_tree):
            adjacency_matrix[i, j] = adjacency_matrix[j, i]
        adjacency_matrix[i, i] = 0
    transition_matrix = build_transition_matrix(adjacency_matrix, alpha, beta)

    s = np.random.uniform(0, H, nb_tree + 1)
    s[nb_tree] = H
    states = [s]
    for i in range(nb_iter):
        s = transition_matrix @ s
        states.append(s)

    plt.plot(states)
    plt.title(
        f"{nb_tree} Trees with random graph (Erdős-Rényi with p={edge_probability}) structure"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Height")
    plt.figure()

    G = graph_from_adjacency_matrix(adjacency_matrix)
    nx.draw(G, with_labels=True)
    # print(adjacency_matrix)
    # print(transition_matrix)

plt.show()


def make_grid_matrix(rows, cols):
    n = rows * cols
    M = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            # Two inner diagonals
            if c > 0:
                M[i - 1, i] = M[i, i - 1] = 1
            # Two outer diagonals
            if r > 0:
                M[i - cols, i] = M[i, i - cols] = 1
    return M


if exp_grid:
    nb_tree = col * row

    adjacency_matrix = make_grid_matrix(row, col)
    transition_matrix = build_transition_matrix(adjacency_matrix, alpha, beta)

    s = np.random.uniform(0, H, nb_tree + 1)
    s[nb_tree] = H

    states = [s]
    for i in tqdm(range(nb_iter), desc="Grid experiment"):
        s = transition_matrix @ s
        states.append(s)

    plt.plot(states)
    plt.title(f"{nb_tree} Trees with {row}x{col} grid structure")
    plt.xlabel("Iteration")
    plt.ylabel("Height")
    plt.figure()

    pos = plt.imshow(
        s[:-1].reshape((row, col)),
        cmap="hot",
        interpolation="nearest",
    )
    plt.colorbar(pos, format="%.2f")
    plt.title("Heightmap of the forest at the end of the simulation")
    # print(adjacency_matrix)

    if grid_plot_asymptotic:
        plt.figure()
        samples_high = []
        samples_low = []
        for _ in range(nb_samples):
            s = np.random.uniform(0, H, nb_tree + 1)
            s[nb_tree] = H
            for i in range(1000):
                s = transition_matrix @ s

            for i in range(nb_tree):
                if s[i] > H:
                    samples_high.append(s[i])
                else:
                    samples_low.append(s[i])
        sns.histplot(samples_high, label="High")
        sns.histplot(samples_low, label="Low")

    plt.show()


def make_octo_grid_matrix(rows, cols):
    n = rows * cols
    M = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            # Two inner diagonals
            if c > 0:
                M[i - 1, i] = M[i, i - 1] = 1
                if r > 0:
                    M[i - cols - 1, i] = M[i, i - cols - 1] = 1
            # Two outer diagonals
            if r > 0:
                M[i - cols, i] = M[i, i - cols] = 1
                if c < cols - 1:
                    M[i - cols + 1, i] = M[i, i - cols + 1] = 1
    return M


if exp_octo_grid:
    nb_tree = col * row

    adjacency_matrix = make_octo_grid_matrix(row, col)
    transition_matrix = build_transition_matrix(adjacency_matrix, alpha, beta)

    s = np.random.uniform(0, H, nb_tree + 1)
    s[nb_tree] = H

    states = [s]
    for i in tqdm(range(nb_iter), desc="Octo grid experiment"):
        s = transition_matrix @ s
        states.append(s)

    plt.plot(states)
    plt.title(f"{nb_tree} Trees with {row}x{col} grid (8 neighbors) structure")
    plt.xlabel("Iteration")
    plt.ylabel("Height")
    plt.figure()

    pos = plt.imshow(
        s[:-1].reshape((row, col)),
        cmap="hot",
        interpolation="nearest",
    )
    plt.title("Heightmap of the forest at the end of the simulation")
    plt.colorbar(pos)
    # print(adjacency_matrix)

    """
    plt.figure()
    G = graph_from_adjacency_matrix(adjacency_matrix)
    nx.draw(G, with_labels=True)
    """

    plt.show()


#######################
# With random cuttings
if exp_random_actions:
    row, col = 3, 3
    nb_tree = col * row
    adjacency_matrix = make_octo_grid_matrix(row, col)
    transition_matrix = build_transition_matrix(adjacency_matrix, alpha, beta)

    s = np.random.uniform(0, H, nb_tree + 1)
    s[nb_tree] = H

    states = [s]
    for i in range(nb_iter):
        s = transition_matrix @ s
        if np.random.rand() < cutting_probability:
            index = np.random.randint(0, nb_tree)
            s[index] = 0
        states.append(s)

    plt.plot(states)
    plt.title(f"{row}x{col} grid (8 neighbors) structure with random cuttings")
    plt.xlabel("Iteration")
    plt.ylabel("Height")
    plt.show()

if exp_cutting_threshold:
    row, col = 3, 3
    nb_tree = col * row
    adjacency_matrix = make_octo_grid_matrix(row, col)
    transition_matrix = build_transition_matrix(adjacency_matrix, alpha, beta)

    s = np.random.uniform(0, H, nb_tree + 1)
    s[nb_tree] = H

    states = [s]
    for i in range(nb_iter):
        s = transition_matrix @ s
        for index in range(nb_tree):
            if s[index] > cutting_threshold:
                s[index] = 0
        states.append(s)

    plt.plot(states)
    plt.title(f"{row}x{col} grid (8 neighbors) structure with cutting threshold")
    plt.xlabel("Iteration")
    plt.ylabel("Height")
    plt.show()
