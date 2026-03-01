#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo II — Location

Produces:
- 2 figures (non-distorted + distorted), each with 4 subpanels:
    (MAE vs gamma1, colored by nu) + (MSE vs gamma1, colored by nu)
    (MAE vs gamma1, colored by eps) + (MSE vs gamma1, colored by eps)

- Table of optimal gamma1* (argmin over gamma1 grid) for gamma2 in {1.0, 1.5, 2.0, 4.0}
  separately for non-distorted and distorted panels, and for nu-axis vs eps-axis.

Key design choices consistent with your LaTeX:
- True DGP uses identity score (equivalently gamma1_true = 2) in the location recursion.
- Innovations are standardized-t (unit variance), or Gaussian when nu = np.inf.
- Contamination is ONLY in estimation sample (train).
- For contaminated eps experiments, the contamination component is conditional on positive outcomes
  (implemented via abs(.) for symmetric distribution).
- Distorted location: out-of-sample omega shifts from 0.0 to 0.1 when eps=1 and nu varies
  (and we apply the same omega shift for the eps-experiments as well, for symmetry; toggle in config).
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from All_models_for_montecarlo import RobustQLEModel


# ======================================================
# CONFIG
# ======================================================

OUTDIR = "MC2_Output_Location"
os.makedirs(OUTDIR, exist_ok=True)

# Samples
T_TRAIN = 2500
T_TEST  = 1000
N_REP   = 3

# True DGP parameters
OMEGA_TRAIN = 0.0
OMEGA_TEST_DISTORTED = 0.1   # distortion applied in test in the distorted panel
GAMMA0 = 0.3                 # your "alpha_0" in LaTeX
BETA0  = 0.1

# Tuning grids
GAMMA1_GRID = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0]   # gamma_1 (alpha_loss)
GAMMA2_LIST = [1.0, 1.5, 2.0, 4.0]                    # evaluation power (table rows)

# Innovation settings
NU_LIST = [3, 5, 8, np.inf]       # nu-axis experiments (eps=1)
NU_FIXED_FOR_EPS = 5              # eps-axis experiments (nu=5)
EPS_LIST = [1, 0.99, 0.98, 0.97]  # eps-axis experiments (nu fixed)

# Mixture scale: contam component std is 10x base component std
STD_RATIO = 10.0

# Keep c fixed throughout (xi=1 in your text; here 'c' plays that role in your code)
C_FIXED = 1.0

# Parallel
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# Distortion policy:
# In your text distortion is described for eps=1, varying nu.
# If you ALSO want distortion for eps-axis experiments, keep this True.
APPLY_DISTORTION_TO_EPS_EXPERIMENTS = True


# ======================================================
# Helpers: standardized t, mixture, conditional positive
# ======================================================

def draw_t_mixture(n, nu, eps, scale_contam=10.0, tilde_nu=None, seed=None):
    """
    Draw eta_t ~ eps * t_nu + (1-eps) * scale_contam * t_tilde_nu
    (RAW Student-t, NOT standardized)
    """
    rng = np.random.default_rng(seed)

    if eps == 1.0:
        return rng.standard_t(nu, size=n)

    if tilde_nu is None:
        tilde_nu = nu  # simplest consistent choice

    base = rng.standard_t(nu, size=n)
    contam = scale_contam * rng.standard_t(tilde_nu, size=n)

    u = rng.random(n)
    return np.where(u < eps, base, contam)


# ======================================================
# DGP: location model with identity score (true gamma1=2)
# ======================================================

def simulate_location_path(
    T_train,
    T_test,
    omega_train,
    omega_test,
    gamma0,
    beta0,
    nu,
    eps_mix_train,
    rng_seed,
    contam_positive_train=False,
):
    rng = np.random.default_rng(rng_seed)

    T = T_train + T_test
    y = np.zeros(T)
    theta = np.zeros(T + 1)

    theta[0] = omega_train / (1 - beta0)

    eta_train = draw_t_mixture(
        T_train, nu, eps_mix_train, rng,
        scale_contam=10.0,
        contam_positive=contam_positive_train
    )
    eta_test = rng.standard_t(nu, size=T_test)  # always clean OOS

    for t in range(T):
        if t < T_train:
            omega = omega_train
            eta = eta_train[t]
        else:
            omega = omega_test
            eta = eta_test[t - T_train]

        y[t] = theta[t] + eta
        theta[t + 1] = omega + gamma0 * (y[t] - theta[t]) + beta0 * theta[t]

    return y



# ======================================================
# Evaluation: "gamma2-loss" as mean |error|^gamma2
# gamma2=1 -> MAE, gamma2=2 -> MSE
# ======================================================

def eval_loss(error: np.ndarray, gamma2: float) -> float:
    return float(np.mean(np.abs(error) ** gamma2))


# ======================================================
# One replication for a given scenario and gamma1
# ======================================================

def one_replication_metrics(
    rep_id: int,
    gamma1: float,
    nu: float,
    eps_mix_train: float,
    distorted: bool,
    axis_tag: str,   # 'nu' or 'eps' (only for metadata)
):
    """
    Returns metrics for all gamma2 in GAMMA2_LIST for a single replication.
    """

    omega_train = OMEGA_TRAIN

    if axis_tag == "nu" and distorted:
        omega_test = OMEGA_TEST_DISTORTED   # 0.1
    else:
        omega_test = OMEGA_TRAIN            # 0.0

    contam_positive_train = (axis_tag == "eps" and distorted and eps_mix_train < 1.0)


    y = simulate_location_path(
        T_train=T_TRAIN,
        T_test=T_TEST,
        omega_train=omega_train,
        omega_test=omega_test,
        gamma0=GAMMA0,
        beta0=BETA0,
        nu=nu,
        eps_mix_train=eps_mix_train,
        rng_seed=1000 + rep_id,
        contam_positive_train=contam_positive_train,  # only for contaminated experiments with fixed nu
        eps_mix_test=1.0,                              # NO contamination in test by design
        contam_positive_test=False
    )

    y_train = y[:T_TRAIN]
    y_test = y[T_TRAIN:]

    model = RobustQLEModel(
        model_type="location",
        alpha_loss=gamma1,   # tuning parameter (fixed)
        c=C_FIXED
    )

    model.fit(y_train)

    # params to array for _filter_volatility
    param_array = np.array([model.params[name] for name in model.param_names])

    # continue filter out-of-sample from last in-sample state
    theta_hat_test = model._filter_volatility(
        y_test,
        param_array,
        f0=model.fitted_volatility[-1]
    )

    err = y_test - theta_hat_test

    out = {"rep": rep_id, "gamma1": gamma1, "nu": nu, "eps": eps_mix_train, "distorted": distorted, "axis": axis_tag}
    for g2 in GAMMA2_LIST:
        out[f"loss_g2_{g2}"] = eval_loss(err, g2)

    return out


# ======================================================
# Run one scenario (fixed nu/eps and distorted flag): compute curves vs gamma1
# ======================================================

def run_scenario(axis_tag: str, axis_value: float, distorted: bool):
    """
    axis_tag = 'nu' : axis_value is nu, eps_train = 1.0
    axis_tag = 'eps': axis_value is eps, nu = NU_FIXED_FOR_EPS
    """
    if axis_tag == "nu":
        nu = axis_value
        eps_train = 1.0
    elif axis_tag == "eps":
        nu = NU_FIXED_FOR_EPS
        eps_train = axis_value
    else:
        raise ValueError("axis_tag must be 'nu' or 'eps'.")

    # For each gamma1, run N_REP replications in parallel and aggregate gamma2 losses
    rows = []
    for gamma1 in GAMMA1_GRID:
        metrics = Parallel(n_jobs=N_JOBS)(
            delayed(one_replication_metrics)(
                rep_id=rep,
                gamma1=gamma1,
                nu=nu,
                eps_mix_train=eps_train,
                distorted=distorted,
                axis_tag=axis_tag,
            )
            for rep in range(N_REP)
        )
        df = pd.DataFrame(metrics)

        agg = {"gamma1": gamma1, "distorted": distorted, "axis": axis_tag}
        if axis_tag == "nu":
            agg["nu"] = nu
            agg["eps"] = 1.0
        else:
            agg["nu"] = NU_FIXED_FOR_EPS
            agg["eps"] = eps_train

        for g2 in GAMMA2_LIST:
            agg[f"mean_loss_g2_{g2}"] = float(df[f"loss_g2_{g2}"].mean())

        rows.append(agg)

    return pd.DataFrame(rows)


# ======================================================
# Plotting helpers
# ======================================================

def plot_mae_mse(df_curves: pd.DataFrame, axis_tag: str, distorted: bool, fname: str):
    title = ("Distorted location" if distorted else "Non-distorted location")
    if axis_tag == "nu":
        group_key = "nu"
        legend_title = r"$\nu$ (eps=1)"
    else:
        group_key = "eps"
        legend_title = r"$\epsilon$ (nu=5)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    for ax, g2, lab in [(axes[0], 1.0, "MAE"), (axes[1], 2.0, "MSE")]:
        for val, g in df_curves.groupby(group_key):
            ax.plot(g["gamma1"], g[f"mean_loss_g2_{g2}"], marker="o", linewidth=1.5, label=str(val))
        ax.set_title(f"{title} — {lab}")
        ax.set_xlabel(r"$\gamma_1$")
        ax.set_ylabel(rf"mean $|e|^{{{g2}}}$")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=legend_title, loc="center right")
    fig.suptitle(f"Location MC — {title} — colored by {legend_title}", y=1.02)
    plt.tight_layout(rect=[0, 0, 0.85, 1.0])

    outpath = os.path.join(OUTDIR, fname)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath



def make_argmin_table(df_all: pd.DataFrame):
    """
    Returns a table with gamma1* for each:
      - distorted in {False, True}
      - axis in {nu, eps}
      - gamma2 in GAMMA2_LIST
      - axis_value in NU_LIST or EPS_LIST
    """
    rows = []
    for distorted in [False, True]:
        for axis_tag in ["nu", "eps"]:
            if axis_tag == "nu":
                axis_values = NU_LIST
                key = "nu"
            else:
                axis_values = EPS_LIST
                key = "eps"

            for g2 in GAMMA2_LIST:
                for axis_val in axis_values:
                    sub = df_all[(df_all["distorted"] == distorted) & (df_all["axis"] == axis_tag) & (df_all[key] == axis_val)]
                    # sub contains multiple gamma1 rows; choose argmin
                    idx = sub[f"mean_loss_g2_{g2}"].idxmin()
                    gamma1_star = float(sub.loc[idx, "gamma1"])
                    rows.append({
                        "distorted": distorted,
                        "axis": axis_tag,
                        key: axis_val,
                        "gamma2": g2,
                        "gamma1_star": gamma1_star
                    })
    return pd.DataFrame(rows)


# ======================================================
# MAIN runner: all location experiments
# ======================================================

def main():
    tasks = []
    # build all scenarios
    # nu-axis: eps=1, vary nu
    for distorted in [False, True]:
        for nu in NU_LIST:
            tasks.append(("nu", nu, distorted))
        for eps in EPS_LIST:
            tasks.append(("eps", eps, distorted))

    total_blocks = len(tasks) * len(GAMMA1_GRID)
    pbar = tqdm(total=total_blocks, desc="Scenario blocks (axis_value × distorted × gamma1)")

    all_curves = []
    for axis_tag, axis_value, distorted in tasks:
        # run scenario (internally parallel over reps)
        df_curves = run_scenario(axis_tag=axis_tag, axis_value=axis_value, distorted=distorted)
        pbar.update(len(GAMMA1_GRID))

        # attach axis_value explicitly for easy grouping later
        if axis_tag == "nu":
            df_curves["nu"] = axis_value
            df_curves["eps"] = 1.0
        else:
            df_curves["nu"] = NU_FIXED_FOR_EPS
            df_curves["eps"] = axis_value

        all_curves.append(df_curves)

    pbar.close()

    df_all = pd.concat(all_curves, ignore_index=True)
    df_all.to_csv(os.path.join(OUTDIR, "location_all_curves.csv"), index=False)

    # Build plots (4 figures total: (nu-axis, non-dist), (nu-axis, dist), (eps-axis, non-dist), (eps-axis, dist))
    plots = []
    for distorted in [False, True]:
        for axis_tag in ["nu", "eps"]:
            sub = df_all[(df_all["distorted"] == distorted) & (df_all["axis"] == axis_tag)].copy()
            fname = f"Location_{'Distorted' if distorted else 'Nondistorted'}_{axis_tag}.png"
            plots.append(plot_mae_mse(sub, axis_tag=axis_tag, distorted=distorted, fname=fname))

    # Argmin table
    df_star = make_argmin_table(df_all)
    df_star.to_csv(os.path.join(OUTDIR, "location_gamma1_star.csv"), index=False)

    # Also print a compact view (pivot) for convenience
    print("\nSaved curves to:", os.path.join(OUTDIR, "location_all_curves.csv"))
    print("Saved argmin table to:", os.path.join(OUTDIR, "location_gamma1_star.csv"))
    print("Saved plots:")
    for p in plots:
        print(" -", p)

    # Example pivot: non-distorted, nu-axis, gamma2=1
    try:
        piv = df_star[(df_star["distorted"] == False) & (df_star["axis"] == "nu") & (df_star["gamma2"] == 1.0)] \
                .pivot(index="gamma2", columns="nu", values="gamma1_star")
        print("\nExample pivot (non-distorted, nu-axis, gamma2=1):")
        print(piv)
    except Exception:
        pass


if __name__ == "__main__":
    main()
