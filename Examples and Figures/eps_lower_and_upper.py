"""
Plot of contamination threshold underline{eps}(gamma_1, alpha) with PRADA upper bound.

- Solid curves:  lower bound underline{eps} from Theorem (robust update dominates gamma_2=2)
- Dashed curves: upper bound bar{eps} such that both updates remain PRADA under P^eps
- Shaded region: valid operating region [underline{eps}, bar{eps}]

DGP: Gaussian location model, MSE evaluation (quadratic scoring rule).
Contamination: H_t = 3 * t_4 (Student-t, 4 df, scale 3), symmetric around theta_pred.
"""


import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['text.usetex'] = True


def psi_barron(x, gamma, xi=1.0):
    if gamma == 2:
        return x / xi**2
    elif gamma == 0:
        return (2 * x) / (x**2 + 2 * xi**2)
    elif gamma == -math.inf:
        return (x / xi**2) * np.exp(-0.5 * (x / xi)**2)
    else:
        return (x / xi**2) * ((x / xi)**2 / abs(gamma - 2) + 1)**((gamma / 2) - 1)


theta_star = 0.0
theta_pred = 1.0
sigma      = 1.0
xi         = 1.0
gamma_2    = 2.0
N          = 500000

np.random.seed(42)
checkx_clean  = np.random.normal(theta_star, sigma, N) - theta_pred
checkx_contam = 3.0 * np.random.standard_t(df=4, size=N)
base = theta_star - theta_pred   # = -1 Note that this is the theoretical value of E(\grad scoring rule) under the clean distribution.

def DP(gamma, alpha, checkx_clean = checkx_clean, checkx_contam = checkx_contam):
    Epsi  = np.mean(psi_barron(checkx_clean, gamma, xi))
    Epsi2 = np.mean(psi_barron(checkx_clean, gamma, xi)**2)
    return 2 * alpha * base * Epsi - alpha**2 * Epsi2

def DH(gamma, alpha, checkx_clean = checkx_clean, checkx_contam = checkx_contam):   
    Epsi  = np.mean(psi_barron(checkx_contam, gamma, xi))
    Epsi2 = np.mean(psi_barron(checkx_contam, gamma, xi)**2)
    return 2 * alpha * base * Epsi - alpha**2 * Epsi2

def compute_prada_upper(gamma_i, alpha, checkx_clean = checkx_clean, checkx_contam =   checkx_contam):
    P = DP(gamma_i, alpha, checkx_clean, checkx_contam)
    H = DH(gamma_i, alpha, checkx_clean, checkx_contam)
    fract = P/(P-H)
    upper_bound = min(fract, 1)
    return upper_bound



def DDP(gamma_1, gamma_2, alpha, checkx_clean = checkx_clean):
    Epsi_1  = np.mean(psi_barron(checkx_clean, gamma_1, xi))
    Epsi2_1 = np.mean(psi_barron(checkx_clean, gamma_1, xi)**2)
    Epsi_2  = np.mean(psi_barron(checkx_clean, gamma_2, xi))
    Epsi2_2 = np.mean(psi_barron(checkx_clean, gamma_2, xi)**2)
    return 2 * alpha * base * (Epsi_1-Epsi_2) - alpha**2 * (Epsi2_1 - Epsi2_2)

def DDH(gamma_1, gamma_2, alpha, checkx_clean = checkx_clean):
    Epsi_1  = np.mean(psi_barron(checkx_contam, gamma_1, xi))
    Epsi2_1 = np.mean(psi_barron(checkx_contam, gamma_1, xi)**2)
    Epsi_2  = np.mean(psi_barron(checkx_contam, gamma_2, xi))
    Epsi2_2 = np.mean(psi_barron(checkx_contam, gamma_2, xi)**2)
    return 2 * alpha * base * (Epsi_1-Epsi_2) - alpha**2 * (Epsi2_1 - Epsi2_2)

def compute_eps_lower(gamma_1, alpha):
    P = DDP(gamma_1, gamma_2, alpha)
    H = DDH(gamma_1, gamma_2, alpha)
    if H <= 0:
        print(False, f"Unexpected case: P={P}, H={H}, alpha={alpha}, gamma_1={gamma_1}")
        return np.nan
    if P >= 0:        
        return 0
    else:             
        return max(-P/(H-P), 0)


g1_values  = [-1.0, 0.0, 0.5, 1.0, 1.5]
alpha_grid = np.linspace(0.001, 0.70, 80)
colors     = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

fig, ax = plt.subplots(figsize=(8, 5))

for g1_val, col in zip(g1_values, colors):
    eps_lb = np.array([compute_eps_lower(g1_val, a) for a in alpha_grid])
    eps_ub_1 = np.array([compute_prada_upper(g1_val, a) for a in alpha_grid])
    eps_ub_2 = np.array([compute_prada_upper(gamma_2, a) for a in alpha_grid])
    eps_ub = np.minimum(eps_ub_1, eps_ub_2)


    ax.plot(alpha_grid, eps_lb, color=col, lw=1, label=rf'$\gamma_1={g1_val}$')
    ax.plot(alpha_grid, eps_ub, color = 'black' , lw=1, ls='--')

    valid = (eps_ub > eps_lb) & np.isfinite(eps_lb)
    ax.fill_between(alpha_grid,
                    np.where(valid, eps_lb, np.nan),
                    np.where(valid, eps_ub, np.nan),
                    color='lightgrey', alpha=0.15)



ax.set_xlim(0, 0.70)
ax.set_ylim(0, 0.70)

ax.plot([], [], 'k--', lw=1.5, label=r'$\bar{\varepsilon}$')
ax.fill_between([], [], [], color='grey', alpha=0.2, label=r'$\varepsilon \in [\underline{\varepsilon}, \bar{\varepsilon})$')

ax.set_xlabel(r'$\alpha$', fontsize=13)
ax.set_ylabel(r'$\varepsilon$', fontsize=13)
ax.legend(fontsize=9, ncol=2, loc='upper right')

plt.tight_layout()

plt.show()