import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""Helper functions for dynamical systems"""

# ggplot colours
RED = "#E24A33"
BLUE = "#348ABD"
GREY = "#777777"


def plot_phase_portrait(system, t_max: float = None, z_0s: list[list[float]] = None,
                        xrange=None, yrange=None, ax=None):
    """Plots phase portrait for system, and trajectories starting at points z_0s for time
    t=0 to t=t_max"""
    if (t_max is None and z_0s is not None) or (t_max is not None and z_0s is None):
        raise Exception(f"t_max and z_0s need to both be set!")
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if xrange is None:
        xrange = [-5, 5]
    if yrange is None:
        yrange = [-5, 5]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    # Plot vector field
    xs, ys = np.meshgrid(np.linspace(*xrange, 20), np.linspace(*yrange, 20))
    Z = system(0, (xs.flatten(), ys.flatten()))
    ax.quiver(xs, ys, *Z, color=BLUE)

    # Plot trajectories
    if z_0s is not None:
        ts = np.linspace(0, t_max, 10**2)
        for z_0 in z_0s:
            sol = solve_ivp(system, t_span=[0, t_max], y0=z_0, dense_output=True)
            z = sol.sol(ts)
            ax.scatter(*z_0, color=RED)
            ax.plot(*z, color=RED)
    return ax


def plot_SIR(beta, nu, mu, t_max, z_0s):
    """Plot SIR trajectories and phase portrait on S-I plane"""
    fig, ax = plt.subplots(1, 2)
    fig.set_tight_layout(True)

    N = 1

    def system(t, z):
        S, I = z
        return [-beta * S * I + mu * (N - S), beta * S * I - nu * I - mu * I]

    xrange = [0, 1]
    yrange = [0, 1]

    # Plot vector field
    n = 20
    xs, ys = np.meshgrid(np.linspace(*xrange, n), np.linspace(*yrange, n))
    # Numpy this. np.flip being weird
    Z = [np.zeros((n, n)), np.zeros((n, n))]
    for i in range(n):
        for j in range(n):
            if xs[i, j] + ys[i, j] <= 1:
                A = system(0, (xs[i, j], ys[i, j]))
                Z[0][i, j] = A[0]
                Z[1][i, j] = A[1]
    ax[0].quiver(xs, ys, *Z, color=BLUE)

    # Plot trajectories
    if z_0s is not None:
        ts = np.linspace(0, t_max, 10**3)
        for i, z_0 in enumerate(z_0s):
            sol = solve_ivp(system, t_span=[0, t_max], y0=z_0, dense_output=True)
            z = sol.sol(ts)
            ax[0].scatter(*z_0, color=RED)
            ax[0].plot(*z, color=RED)

            ax[1].plot(ts, z[0], label=f"Susceptible (traj {i+1})")
            ax[1].plot(ts, z[1], label=f"Infected (traj {i+1})")

    ax[0].set_title("Phase portrait")
    ax[1].set_title("Population")
    ax[1].legend()
    ax[0].set_xlabel("Susceptible")
    ax[0].set_ylabel("Infected")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Count")
    return
