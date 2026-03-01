import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

eval_labels = ["MAE", "Barron 1.5", "MSE", "Barron 4", "Loglik"]

eps_configs = {
    "1.00": {
        "file": "/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps1/mLoss_eps1_dNu10_Eps1.0.npy",
        "grid": np.linspace(-4, 2.5, 200)
    },
    "0.99": {
        "file": "/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps0.99/mLoss_eps0.99_dNu10_Eps0.99.npy",
        "grid": np.linspace(-4, 2.1, 102)
    },
    "0.98": {
        "file": "/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps0.98/mLoss_eps0.98_dNu10_Eps0.98.npy",
        "grid": np.linspace(-4, 2.0, 100)
    },
    "0.97": {
        "file": "/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps0.97/mLoss_eps0.97_dNu10_Eps0.97.npy",
        "grid": np.linspace(-4, 2.0, 100)
    }
}

# ============================================================
# TABLE STORAGE
# ============================================================

OptimalValueTable = pd.DataFrame(index=eval_labels)
GammaStarTable = pd.DataFrame(index=eval_labels)

# ============================================================
# LOOP OVER EPS
# ============================================================

for eps_label, config in eps_configs.items():

    LossMat = np.load(config["file"])
    MeanLoss = np.nanmean(LossMat, axis=2)
    gamma_grid = config["grid"]

    blade_opt_vals = []
    blade_gamma_stars = []
    beta_t_vals = []

    for i, label in enumerate(eval_labels):

        curve = MeanLoss[i, :-1]      # BLADE models
        beta_t_value = MeanLoss[i, -1]  # Beta-t GARCH (last column)

        if label == "Loglik":
            idx = np.nanargmax(curve)
        else:
            idx = np.nanargmin(curve)

        blade_opt_vals.append(curve[idx])
        blade_gamma_stars.append(gamma_grid[idx])
        beta_t_vals.append(beta_t_value)

    # Add two columns per epsilon
    OptimalValueTable[f"BLADE ($\\varepsilon={eps_label}$)"] = blade_opt_vals
    OptimalValueTable[f"$\\beta$-t ($\\varepsilon={eps_label}$)"] = beta_t_vals

    GammaStarTable[f"BLADE ($\\varepsilon={eps_label}$)"] = blade_gamma_stars
    GammaStarTable[f"$\\beta$-t ($\\varepsilon={eps_label}$)"] = [np.nan]*len(eval_labels)

# ============================================================
# PRINT TABLES
# ============================================================

print("\n=== Optimal Loss / LogLik Values ===")
print(OptimalValueTable.round(4))

print("\n=== Corresponding Optimal Gamma1 ===")
print(GammaStarTable.round(3))  



import numpy as np
import pandas as pd

eval_labels = ["MAE", "Barron 1.5", "MSE", "Barron 4", "Loglik"]

eps_configs = {
    "1.00": "/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps1/mLoss_eps1_dNu10_Eps1.0.npy",
    "0.99": "/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps0.99/mLoss_eps0.99_dNu10_Eps0.99.npy",
    "0.98": "/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps0.98/mLoss_eps0.98_dNu10_Eps0.98.npy",
    "0.97": "/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps0.97/mLoss_eps0.97_dNu10_Eps0.97.npy"
}

BetaTTable = pd.DataFrame(index=eval_labels)

for eps_label, file_path in eps_configs.items():

    LossMat = np.load(file_path)
    MeanLoss = np.nanmean(LossMat, axis=2)

    # Last column = Beta-t GARCH
    beta_values = MeanLoss[:, -1]

    BetaTTable[f"$\\varepsilon = {eps_label}$"] = beta_values

print("\n=== Beta-t GARCH Loss Values ===")
print(BetaTTable.round(4))


# ============================================================
# RELATIVE IMPROVEMENT VS GAMMA_1 = 2
# ============================================================

RelImprovementTable = pd.DataFrame(index=eval_labels)

for eps_label, file_path in eps_configs.items():

    LossMat = np.load(file_path)
    MeanLoss = np.nanmean(LossMat, axis=2)

    # Manually reconstruct gamma grid (same as before)
    if eps_label == "1.00":
        gamma_grid = np.linspace(-4, 2.5, 200)
    elif eps_label == "0.99":
        gamma_grid = np.linspace(-4, 2.1, 102)
    elif eps_label == "0.98":
        gamma_grid = np.linspace(-4, 2.0, 100)
    elif eps_label == "0.97":
        gamma_grid = np.linspace(-4, 2.0, 100)

    # Find gamma_1 = 2
    gamma2_idx = np.argmin(np.abs(gamma_grid - 2.0))

    rel_vals = []

    for i, label in enumerate(eval_labels):

        curve = MeanLoss[i, :-1]  # exclude beta-t

        loss_gamma2 = curve[gamma2_idx]

        if label == "Loglik":
            idx = np.nanargmax(curve)
        else:
            idx = np.nanargmin(curve)

        loss_opt = curve[idx]

        rel_improve = 100 * (loss_gamma2 - loss_opt) / loss_gamma2
        rel_vals.append(rel_improve)

    RelImprovementTable[f"$\\varepsilon={eps_label}$"] = rel_vals

RelImprovementTable = RelImprovementTable.round(2)

print("\n=== Relative Improvement (%) vs γ₁ = 2 ===")
print(RelImprovementTable)