"""
Convergence dynamics of the canonical BLADE(1,1) update.

With omega = 0 and beta = 1, the update reduces to

    vartheta_{t+1|t} = vartheta_{t|t-1}
                       + alpha * psi^B_{gamma, xi}(y_t - vartheta_{t|t-1}),

where psi^B_{gamma, xi} is the derivative of the Barron loss. This script
shows the trajectories starting from several initial values vartheta_0
with a constant observation y_t = y.
"""

import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


# ---------------------------------------------------------------------------
# Matplotlib / LaTeX setup
# ---------------------------------------------------------------------------

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": "15"})
rc("text", usetex=True)


# ---------------------------------------------------------------------------
# Barron loss derivative
# ---------------------------------------------------------------------------

def barron_loss_derivative(x, gamma, xi=1.0):
    r"""Derivative psi^B_{gamma, xi}(x) = d/dx L^B_{gamma, xi}(x).

    For gamma not in {-inf, 0, 2}:

        psi^B_{gamma, xi}(x) = (x / xi^2)
                               * (x^2 / (xi^2 |gamma - 2|) + 1)^{gamma/2 - 1}.
    """
    if gamma == 2:
        return x / xi ** 2
    if gamma == 0:
        return (2.0 * x) / (x ** 2 + 2.0 * xi ** 2)
    if gamma == -math.inf:
        return (x / xi ** 2) * np.exp(-0.5 * (x / xi) ** 2)

    abs_gm2 = np.abs(gamma - 2)
    return (x / xi ** 2) * ((x / xi) ** 2 / abs_gm2 + 1.0) ** (gamma / 2 - 1)


# ---------------------------------------------------------------------------
# BLADE(1,1) trajectory
# ---------------------------------------------------------------------------

def blade_trajectory(vartheta_0, y_obs, gamma, alpha, xi=1.0, steps=20):
    r"""Iterate the canonical BLADE(1,1) update (omega = 0, beta = 1)."""
    vartheta = [vartheta_0]
    for _ in range(steps):
        residual = y_obs - vartheta[-1]
        vartheta.append(vartheta[-1] + alpha * barron_loss_derivative(residual, gamma, xi))
    return vartheta


def plot_convergence(inits, y_obs, gamma, alpha, xi=1.0, steps=20):
    """Plot BLADE trajectories from several initial values vartheta_0."""
    fig, ax = plt.subplots()
    for vartheta_0 in inits:
        traj = blade_trajectory(vartheta_0, y_obs, gamma, alpha, xi, steps)
        ax.plot(traj, "-o", label=rf"$\vartheta_0 = {vartheta_0}$")

    ax.hlines(y_obs, 0, steps, linestyles="dashed", colors="k", label=r"$y$")
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$\vartheta$")
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(False)
    ax.axhline(0, color="0.8", linewidth=1)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_convergence(
        inits=[-5, -2, 0, 5, 10],
        y_obs=2.0,
        gamma=0.5,
        alpha=0.2,
        xi=1.0,
        steps=20,
    )

    plt.show()
