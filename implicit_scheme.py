import numpy as np
from scipy.linalg import solve_banded
from tqdm import tqdm


"""
Potential equal to V0 within [xv, xv+h] and 0 outside
"""
def Pi_like_potential(x, xv, h, V0):
    return np.where((x > xv) & (x < xv + h), V0, 0)


def banded_matrix(V, dt, dx):
    """
    Construct banded matrix for Schrodinger equation,
    given potential V, time step dt, coordinate step dx
    """
    Nx = V.shape[0]
    gamma = 1j * dt / (2 * dx**2)
    lower = -gamma * np.ones(Nx-1)
    main = (1 + 2*gamma + 1j * V * dt).copy()
    upper = -gamma * np.ones(Nx-1)
    main[0], main[-1] = 1, 1
    lower[-1], upper[0] = 0, 0
    ab = np.zeros((3, Nx), dtype=complex)
    ab[0, 1:] = upper
    ab[1, :] = main
    ab[2, :-1] = lower
    return ab


def time_step(psi, ab):
    """
    Evolve wavefunction by 1 time step
    """
    return solve_banded((1, 1), ab, psi)


def solve(psi0, x, V0, h, xv, Nt, dt):
    """
    Evolve wavefunction by time Nt * dt
    Return history in x basis
    """
    V = Pi_like_potential(x, xv, h, V0)

    dx = x[1] - x[0]
    ab = banded_matrix(V, dt, dx)
    psi = psi0.copy()

    history = np.zeros((Nt+1, len(psi)), dtype=complex)
    history[0] = psi0

    for i in tqdm(range(Nt), desc="Solving time evolution"):
        psi = time_step(psi, ab)
        history[i+1] = psi / np.sqrt(np.sum(np.abs(psi**2)) * dx)

    return history
