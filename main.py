import numpy as np
import matplotlib.pyplot as plt
from fourier import solve as fourier_solve
import matplotlib.animation as animation
from implicit_scheme import solve as implicit_solve
from tqdm import tqdm

"""
Approximate integral \int |psi(x)|^2 dx 
by Riemann sum: \sum_i |psi(x_i)|^2 \delta x
Return sqrt of norm mentioned above
"""
def grid_norm(psi, x):
    dx = x[1] - x[0]
    return np.sqrt(np.sum(np.abs(psi**2)) * dx)

"""
Initialize Gaussian wavepacked with initial position x0, initial momentum k0, width sigma
Normalize psi so that \sum_i |psi(x_i)|^2 \delta x = 1
"""
def init_wavepacket(x, x0, k0, sigma):
    psi = np.exp(-(x-x0)**2 / (4 * sigma ** 2))
    psi = psi * np.exp(1j * k0 * x) / grid_norm(psi, x)
    return psi

# Parameters
L = 10.0  # Domain length
V0 = 10.0  # Barrier height
h = 1  # Barrier width
xv = 4.5  # Barrier start
dt = 0.005
T = 1
# Gaussian wave packet parameters
x0 = 1  # Initial packet center
sigma = .1  # Initial packet width
k0 = 50
# Maximum energy state represented in fourier domain
nmax = 1000

generate_gif = True

Nt = int(T / dt)
# Note: if space discretization step is dx, then fourier domain bounds are [-pi / dx, + pi /dx]
# Since k_max = (pi * nmax / L + k0), we have to ensure pi/dx > (k0 + pi * nmax / L)
dx = 1 / (k0 / np.pi + nmax / L)
print(dx)
Nx = int(L / dx) + 1
x = np.linspace(0, L, Nx)
psi0 = init_wavepacket(x, x0, k0, sigma)

print("Starting time evolution for implicit scheme method...")
history_implicit = implicit_solve(psi0, x, V0, h, xv, Nt, dt)

print("Starting time evolution for fourier method...")
history_fourier, c_history = fourier_solve(psi0, x, V0, h, xv, Nt, dt, nmax)

max_value = max(np.max(np.abs(history_fourier)), np.max(np.abs(history_implicit)))
fig, axs = plt.subplots(3, 1, figsize=(8, 12))


line1, = axs[0].plot([], [], lw=2, label='Method 1 Im')
line2, = axs[0].plot([], [], lw=2, label='Method 2 Im')

line3, = axs[1].plot([], [], lw=2, label='Method 1 Re')
line4, = axs[1].plot([], [], lw=2, label='Method 2 Re')

line5, = axs[2].plot([], [], lw=2, label='Method 1 Abs')
line6, = axs[2].plot([], [], lw=2, label='Method 2 Abs')

for ax in axs:
    ax.set_xlim(0, L)
    ax.set_ylim(-max_value, max_value)
    ax.set_xlabel("x")
    ax.axvline(xv, ls=':', label="Potential borders")
    ax.axvline(xv + h, ls=':')
    ax.legend()

if generate_gif:
    print("Rendering GIF...")
    ani = animation.FuncAnimation(fig, lambda frame: [
        line1.set_data(x, history_implicit[frame].imag),
        line2.set_data(x, history_fourier[frame].imag),
        line3.set_data(x, history_implicit[frame].real),
        line4.set_data(x, history_fourier[frame].real),
        line5.set_data(x, np.abs(history_implicit[frame])),
        line6.set_data(x, np.abs(history_fourier[frame])),
    ], frames=tqdm(range(len(history_fourier))), interval=1, blit=False)
    ani.save("free_evolution_then_bounce.gif", writer="pillow", dpi=50)
    plt.show()
else:
    plt.ion()
    print("Start Rendering...")
    frames = tqdm(range(len(history_fourier)), desc="Rendering")
    for frame in frames:
        current_t = frame * dt 
        
        norm1 = grid_norm(history_implicit[frame], x)
        line1.set_data(x, history_implicit[frame].imag)
        line3.set_data(x, history_implicit[frame].real)
        line5.set_data(x, np.abs(history_implicit[frame]))

        norm2 = grid_norm(history_fourier[frame], x)
        line2.set_data(x, history_fourier[frame].imag)
        line4.set_data(x, history_fourier[frame].real)
        line6.set_data(x, np.abs(history_fourier[frame]))

        axs[0].set_title(
            f"Wave Packet Evolution (t = {current_t:.4f}), norm1= {norm1:.4f}, norm2 = {norm2:.4f}")
        plt.draw()
        plt.pause(0.001)

    plt.ioff()
    plt.show()