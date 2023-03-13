# Try to solve the LinearForestEnv as a LQR problem

import numpy as np
import scipy as sp
from scipy.linalg import solve, eigvals
from linear_dynamic_env import ForestLinearEnv
from utils import make_grid_matrix


def dlqr(A, B, Q, R, N=None):
    """Solve the discrete time lqr controller by solving the discrete algebraic Ricatti equation."""
    R = np.eye(B.shape[1]) if R is None else np.array(R, ndmin=2)

    S = sp.linalg.solve_discrete_are(A, B, Q, R, e=None, s=N)
    if N is None:
        K = solve(B.T @ S @ B + R, B.T @ S @ A)
    else:
        K = solve(B.T @ S @ B + R, B.T @ S @ A + S.T)
    return K, S


row, col = 3, 3
nb_iter = 100
nb_run = 10
H = 20

nb_tree = row * col
adjacency_matrix = make_grid_matrix(row, col)
env = ForestLinearEnv(nb_tree, adjacency_matrix, H=H, alpha=0.2, beta=0.1)

if __name__ == "__main__":
    Q = np.zeros((env.n_tree + 1, env.n_tree + 1))
    R = np.eye(env.n_tree)
    N = np.zeros((env.n_tree + 1, env.n_tree))
    K, S = dlqr(env.A, env.B, env.M, env.N)
    print(K)
    print(S)
