"""
Barron loss function and its derivative.

Reproduces panels (a) and (b) of Figure 1 in the BLADE paper:

    (a) L^B_{gamma, xi}(x) for several values of gamma.
    (b) psi^B_{gamma, xi}(x) = d/dx L^B_{gamma, xi}(x) for the same values.

Limit cases handled explicitly:
    gamma =  2        ->  quadratic (L2) loss
    gamma =  1        ->  Charbonnier (smoothed L1) loss
    gamma =  0        ->  Cauchy loss
    gamma = -2        ->  Geman-McClure loss
    gamma = -infty    ->  Welsch loss
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
# Barron loss and its derivative
# ---------------------------------------------------------------------------

def barron_loss(x, gamma, xi=1.0):
    r"""Barron loss L^B_{gamma, xi}(x).

    For gamma not in {-inf, 0, 2}:

        L^B_{gamma, xi}(x) = (|gamma - 2| / gamma)
                             * ( (x^2 / (xi^2 |gamma - 2|) + 1)^{gamma/2} - 1 ).
    """
    if gamma == 2:
        return 0.5 * (x / xi) ** 2
    if gamma == 0:
        return np.log(0.5 * (x / xi) ** 2 + 1.0)
    if gamma == -math.inf:
        return 1.0 - np.exp(-0.5 * (x / xi) ** 2)

    abs_gm2 = np.abs(gamma - 2)
    return (abs_gm2 / gamma) * (
        ((x / xi) ** 2 / abs_gm2 + 1.0) ** (gamma / 2) - 1.0
    )


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
# Plotting helpers
# ---------------------------------------------------------------------------

def _gamma_label(gamma):
    if gamma == -math.inf:
        return r"$\gamma = -\infty$"
    return rf"$\gamma = {gamma}$"


def _style_axes(ax):
    ax.grid(False)
    ax.axhline(0, color="0.8", linewidth=1)


def plot_barron_loss(gammas, xi=1.0, x_range=(-6, 6), n_points=400):
    """Plot L^B_{gamma, xi}(x) for each gamma in `gammas`."""
    x = np.linspace(x_range[0], x_range[1], n_points)

    fig, ax = plt.subplots()
    for gamma in gammas:
        ax.plot(x, barron_loss(x, gamma, xi), label=_gamma_label(gamma))

    ax.set_title(r"Barron loss function for various $\gamma$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$L_{\gamma, \xi}^{\mathrm{B}}(x)$")
    ax.set_ylim(0, 5)
    ax.legend(fontsize=12)
    _style_axes(ax)
    return fig


def plot_barron_derivative(gammas, xi=1.0, x_range=(-6, 6), n_points=400):
    """Plot psi^B_{gamma, xi}(x) for each gamma in `gammas`."""
    x = np.linspace(x_range[0], x_range[1], n_points)

    fig, ax = plt.subplots()
    for gamma in gammas:
        ax.plot(x, barron_loss_derivative(x, gamma, xi), label=_gamma_label(gamma))

    ax.set_title(r"Derivative of Barron loss function for various $\gamma$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\psi_{\gamma, \xi}^{\mathrm{B}}(x)$")
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=12)
    _style_axes(ax)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GAMMAS = [-math.inf, -2, 0, 0.5, 1, 1.5, 2]
    XI = 1.0

    plot_barron_loss(GAMMAS, xi=XI)
    plot_barron_derivative(GAMMAS, xi=XI)

    plt.show()
