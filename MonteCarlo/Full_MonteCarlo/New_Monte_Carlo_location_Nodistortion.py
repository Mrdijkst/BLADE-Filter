#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo: Location model
Non-distorted location, no contamination

Uses RobustQLEModel exactly as implemented in All_models.py
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from All_models_for_montecarlo import RobustQLEModel


# ======================================================
# 1. DGP: location model
# ======================================================

def simulate_location_dgp(T, omega, gamma, beta, nu, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.zeros(T)
    theta = np.zeros(T + 1)

    theta[0] = omega / (1 - beta)
    eps = np.random.standard_t(nu, size=T)

    for t in range(T):
        y[t] = theta[t] + eps[t]
        theta[t + 1] = omega + gamma * (y[t] - theta[t]) + beta * theta[t]

    return y


# ======================================================
# 2. One replication
# ======================================================

def run_replication(rep, gamma_1, nu, T_train, T_test):
    # --- true parameters ---
    omega0, gamma0, beta0 = 0.0, 0.3, 0.1

    y = simulate_location_dgp(
        T=T_train + T_test,
        omega=omega0,
        gamma=gamma0,
        beta=beta0,
        nu=nu,
        seed=1000 + rep
    )

    y_train = y[:T_train]
    y_test = y[T_train:]

    # --------------------------------------------------
    # Estimate location model with FIXED gamma_1
    # --------------------------------------------------
    model = RobustQLEModel(
        model_type='location',
        alpha_loss=gamma_1,   # <-- γ₁ is fixed
        c=1.0
    )

    model.fit(y_train)

    # --------------------------------------------------
    # Extract filtered location
    # --------------------------------------------------
    param_array = np.array([model.params[name] for name in model.param_names])

    theta_hat_test = model._filter_volatility(
        y_test,
        param_array,
        f0=model.fitted_volatility[-1]
    )


    mae = np.mean(np.abs(y_test - theta_hat_test))

    return mae


# ======================================================
# 3. Run Monte Carlo
# ======================================================

def run_mc():
    nu = 5
    T_train = 2500
    T_test = 1000
    n_rep = 3

    gamma_grid = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0]

    n_jobs = multiprocessing.cpu_count() - 1  # leave 1 core free
    rows = []

    total_runs = len(gamma_grid) * n_rep

    with tqdm(total=total_runs, desc="Monte Carlo replications") as pbar:

        for gamma_1 in gamma_grid:

            # --- parallel replications for fixed gamma_1 ---
            maes = Parallel(n_jobs=n_jobs)(
                delayed(run_replication)(
                    rep, gamma_1, nu, T_train, T_test
                )
                for rep in range(n_rep)
            )

            # update progress bar *after* reps finished
            pbar.update(n_rep)

            rows.append({
                "gamma_1": gamma_1,
                "mean_MAE": np.mean(maes)
            })

    return pd.DataFrame(rows)





# ======================================================
# 4. Main
# ======================================================

if __name__ == "__main__":
    df = run_mc()
    print(df)

    gamma_star = df.loc[df["mean_MAE"].idxmin(), "gamma_1"]
    print(f"\nOptimal gamma_1 (MAE): {gamma_star}")
