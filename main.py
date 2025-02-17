import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fourier import solve as fourier_solve
from implicit_scheme import solve as implicit_solve
from tqdm import tqdm
from scipy.integrate import simpson

"""
Approximate integral \int |psi(x)|^2 dx 
by Riemann sum: \sum_i |psi(x_i)|^2 \delta x = 1
Return sqrt of norm mentioned above
"""
def grid_norm(psi, x):
    dx = x[1] - x[0]
    return np.sqrt(np.sum(np.abs(psi**2)) * dx)

"""
Initialize Gaussian wavepacked with initial position x0, initial speed v0, width sigma
Normalize psi so that \sum_i |psi(x_i)|^2 \delta x = 1
"""
def init_wavepacket(x, x0, v0, sigma):
    psi = np.exp(-(x-x0)**2 / (2 * sigma ** 2))
    psi = psi * np.exp(1j * v0 * x) / grid_norm(psi, x)
    return psi


# Parameters
L = 60.0  # Domain length
V0 = 110.0  # Barrier height
h = .5  # Barrier width
xv = 30  # Barrier start

dt = 0.0005
T = .7
Nt = int(T / dt)

# Gaussian wave packet parameters
x0 = 28  # Initial packet center
sigma = .1  # Initial packet width
v0 = 20

#Maximum energy state represented in fourier domain
nmax = 1000

# Note: if space disctetization step is dx, then fourier domain bounds are [-pi / dx, + pi /dx]
# Since k_max = pi * nmax / L, we have to ensure pi/dx > pi * nmax / L, or dx < L / nmax
dx = L / (nmax+1000) #1000 for extra precision
print(dx)
Nx = int(L / dx) + 1
x = np.linspace(0, L, Nx)
psi0 = init_wavepacket(x, x0, v0, sigma)

print("Starting time evolution for implicit scheme method...")
history_implicit = implicit_solve(psi0, x, V0, h, xv, Nt, dt)

print("Starting time evolution for fourier method...")
history_fourier, c_history = fourier_solve(psi0, x, V0, h, xv, Nt, dt, nmax)

max_value = max(np.max(np.abs(history_fourier)), np.max(np.abs(history_implicit)))
fig, ax = plt.subplots()
ax.set_xlim(20, 40)
ax.set_ylim(-max_value, max_value)
ax.set_xlabel("x")
ax.set_title("Wave Packet Evolution")
line1, = ax.plot([], [], lw=2, label='Method 1')
line2, = ax.plot([], [], lw=2, label='Method 2')
ax.axvline(xv, ls=':', label="Potential borders")
ax.axvline(xv + h, ls=':')
ax.legend()

def update(frame):
    current_t = frame * dt 
    
    norm1 = grid_norm(history_implicit[frame], x)
    line1.set_data(x, history_implicit[frame].imag)

    norm2 = grid_norm(history_fourier[frame], x)
    line2.set_data(x, history_fourier[frame].imag)

    #keep track of wavefunction norm (must be equal to 1)
    ax.set_title(
        f"Wave Packet Evolution (t = {current_t:.4f}), norm1= {norm1:.4f}, norm2 = {norm2:.4f}")
    return line1, line2


print("Start Rendering...")
frames = tqdm(range(len(history_fourier)), desc="Rendering")
ani = animation.FuncAnimation(fig, update, frames, interval=50, blit=False)
ani.save("wavepacket.gif", writer="pillow", dpi=50)
plt.show()
