import numpy as np
from scipy.linalg import expm
from tqdm import tqdm


def k_n(n, L):
    """
    wave number
    """
    return np.pi * n / L

def E_n(n, L):
    """
    energy a.k.a eigenvalue
    """
    return k_n(n, L)**2 / 2

def eigenfunctions(x, nmax, L):
    """
    Return eigenfunctions as array of shape (nmax, x.shape[0]).
    wave_funcs[i] is i-th eigenfunction for i = 1, ..., nmax
    """
    n_values = np.arange(1, nmax + 1)
    k_n_values = k_n(n_values[..., np.newaxis], L)
    return np.sqrt(2 / L) * np.sin(k_n_values * x[np.newaxis, ...])

def H_nbasis(nmax, L, V0, h, xv):
    """
    Hamiltonian in |n> basis
    """
    indices = np.arange(1, nmax+1, dtype=complex)
    n, m = np.meshgrid(indices, indices)

    pi_diff = np.pi * (m - n + 1e-10)
    pi_sum = np.pi * (m + n)

    H = np.zeros((nmax, nmax), dtype=complex)
    np.fill_diagonal(H, E_n(indices, L))

    def F(x):
        return np.sin(pi_diff * x / L) / pi_diff - np.sin(pi_sum * x / L) / pi_sum

    H = H + V0 * (F(xv+h) - F(xv))

    return H

def x_to_n_transform(psi_x, eigenfuncs, dx):
    """
    Transform wave function from x to |n> basis representation.
    Return vector of c_i such that psi_x = \sum c_i * phi_i,
    where phi_i are eigenfunctions
    """
    #approximate integral with Riemann sum
    return np.dot(eigenfuncs, psi_x) * dx

def n_to_x_transform(c, eigenfuncs):
    """
    Transform wave function from basis to position representation.
    Return wave function in position space.
    """
    return np.dot(c, eigenfuncs)

def solve(psi_0, x, V0, h, xv, N_t, dt, nmax=100):
    """
    Evolve wavefunction by time Nt * dt
    Return history in x basis and |n> basis
    """
    L = x[-1] - x[0]
    dx = x[1] - x[0]
    
    eigenfuncs = eigenfunctions(x, nmax, L) 
    H = H_nbasis(nmax, L, V0, h, xv)
    c = x_to_n_transform(psi_0, eigenfuncs, dx)
    U = expm(-1j * H * dt)
    
    history = np.zeros((N_t+1, psi_0.shape[0]), dtype=complex)
    c_history = np.zeros((N_t+1, c.shape[0]), dtype=complex)
    history[0] = psi_0
    c_history[0] = c
    
    for i in tqdm(range(N_t), desc="Solving time evolution"):
        c = np.dot(U, c)
        history[i+1] = n_to_x_transform(c, eigenfuncs)
        c_history[i+1] = c
    
    return history, c_history
