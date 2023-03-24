import numpy as np


def make_grid_matrix(rows, cols):
    """
    Make adjacency matrix for a grid graph (4 neighbors)

    Parameters
    ----------
    rows : int
        Number of rows
    cols : int
        Number of columns

    Returns
    -------
    M : np.array
        Adjacency matrix of the grid graph
    """
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


def make_octo_grid_matrix(rows, cols):
    """
    Make adjacency matrix for a grid graph (8 neighbors)

    Parameters
    ----------
    rows : int
        Number of rows
    cols : int
        Number of columns

    Returns
    -------
    M : np.array
        Adjacency matrix of the grid graph
    """
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


def build_transition_matrix(adjacency_matrix, alpha, beta):
    """
    Build transition matrix for a linear dynamic forest environment

    Parameters
    ----------
    adjacency_matrix : np.array
        Adjacency matrix of the graph
    alpha : float
        Growth parameter (0 <= alpha <= 1, influence of the growth rate of the trees)
    beta : float
        Interaction parameter (0 <= beta <= 1, influence of the interaction between trees)

    Returns
    -------
    transition_matrix : np.array
        Transition matrix of the linear dynamic forest environment
    """

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


def make_random_graph_matrix(n, p):
    """
    Make adjacency matrix for a simple random Erdos-Renyi graph

    Parameters
    ----------
    n : int
        Number of nodes
    p : float
        Probability of an edge between two nodes
    """
    adjacency_matrix = np.random.binomial(1, p, (n, n))
    for i in range(n):  # Make it symmetric
        for j in range(i, n):
            adjacency_matrix[i, j] = adjacency_matrix[j, i]
        adjacency_matrix[i, i] = 0
