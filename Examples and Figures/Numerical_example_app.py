"""
Numerical computations for Examples  illustrating Theorem 1 and 2.

Example 1: clean dominance under P_t = t_4(0,1)
Example 2: contaminated case with P_t = N(0,1), H_t = 3*t_4
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import t as student_t, norm

# ---------- Setup ----------
# Fixed parameters
THETA_PRED = 1.0   # vartheta_{t|t-1}
XI = 1.0           # Barron scale parameter
GAMMA_1 = 1.0      # robust update
GAMMA_2 = 2.0      # non-robust (quadratic) update
C = 2.0            # |E[H_S]| bound; for S = -(y-theta)^2, H_S = -2 exactly


def pdf_t4(y, scale=1.0):
    """Student-t with 4 df, location 0, given scale."""
    return student_t.pdf(y, df=4, loc=0.0, scale=scale)

def pdf_normal(y):
    """Standard normal."""
    return norm.pdf(y, loc=0.0, scale=1.0)


def psi_gamma_1(y):
    """Barron score for gamma=1, xi=1: pseudo-Huber form."""
    x = y - THETA_PRED
    return x / np.sqrt(x * x + 1.0)

def psi_gamma_2(y):
    """Barron score for gamma=2, xi=1: identity (Gaussian score)."""
    return y - THETA_PRED


def moments(psi, pdf):
    """Compute E[psi] and E[psi^2] under the given density."""
    m, _ = quad(lambda y: psi(y) * pdf(y), -np.inf, np.inf)  
    v, _ = quad(lambda y: psi(y) ** 2 * pdf(y), -np.inf, np.inf)
    return m, v

# For S = -(y-theta)^2, exact second-order expansion gives
#   Delta D_S(phi_i) = -2*alpha*E[psi_i] - alpha^2*E[psi_i^2]
def delta_D(alpha, m, v):
    return -2.0 * alpha * m - alpha ** 2 * v

# ============================================================
# EXAMPLE 1:  P_t = t_4(0,1), clean dominance
# ============================================================
print("=" * 60)
print("Example 1:  P_t = t_4(0,1)")
print("=" * 60)

m1_P, v1_P = moments(psi_gamma_1, lambda y: pdf_t4(y, scale=1))
m2_P, v2_P = moments(psi_gamma_2, lambda y: pdf_t4(y, scale=1))

print(f"  E_P[psi_1]      = {m1_P:+.4f}")
print(f"  E_P[psi_1^2]    = {v1_P:+.4f}")
print(f"  E_P[psi_2]      = {m2_P:+.4f}   (exact: -1)")
print(f"  E_P[psi_2^2]    = {v2_P:+.4f}   (exact: 3)")

# Thresholds
# bar_alpha(i) = (2/c) * E[grad S] * E[psi_i] / E[psi_i^2]
#              = (2/2) * (-2)*m_i / v_i = -2*m_i / v_i
bar_alpha_1 = -2.0 * m1_P / v1_P
bar_alpha_2 = -2.0 * m2_P / v2_P
bar_alpha = min(bar_alpha_1, bar_alpha_2)

# underline_alpha = E[grad S] * (E[psi_2] - E[psi_1]) / (E[psi_2^2] - E[psi_1^2])
#                 = -2*(m2 - m1) / (v2 - v1)
under_alpha = -2.0 * (m2_P - m1_P) / (v2_P - v1_P)

print(f"\n  underline_alpha = {under_alpha:.4f}")
print(f"  bar_alpha(g_1)  = {bar_alpha_1:.4f}")
print(f"  bar_alpha(g_2)  = {bar_alpha_2:.4f}")
print(f"  bar_alpha       = {bar_alpha:.4f}")
print(f"  Interval        = [{under_alpha:.4f}, {bar_alpha:.4f})")

# ============================================================
# EXAMPLE 2:  P_t = N(0,1), H_t = 3*t_4, contaminated
# ============================================================
print("\n" + "=" * 60)
print("Example 2:  P_t = N(0,1),  H_t = 3*t_4")
print("=" * 60)

# Moments under P_t = N(0,1)
m1_P2, v1_P2 = moments(psi_gamma_1, pdf_normal)
m2_P2, v2_P2 = moments(psi_gamma_2, pdf_normal)

# Moments under H_t = 3*t_4
m1_H, v1_H = moments(psi_gamma_1, lambda y: pdf_t4(y, scale=3.0))
m2_H, v2_H = moments(psi_gamma_2, lambda y: pdf_t4(y, scale=3.0))

print("Under P_t = N(0,1):")
print(f"  E_P[psi_1]      = {m1_P2:+.4f}")
print(f"  E_P[psi_1^2]    = {v1_P2:+.4f}")
print(f"  E_P[psi_2]      = {m2_P2:+.4f}   (exact: -1)")
print(f"  E_P[psi_2^2]    = {v2_P2:+.4f}   (exact: 2)")

print("\nUnder H_t = 3*t_4:")
print(f"  E_H[psi_1]      = {m1_H:+.4f}")
print(f"  E_H[psi_1^2]    = {v1_H:+.4f}")
print(f"  E_H[psi_2]      = {m2_H:+.4f}   (exact: -1)")
print(f"  E_H[psi_2^2]    = {v2_H:+.4f}   (exact: 19)")

# Compute thresholds at chosen alpha
ALPHA = 0.5
print(f"\nAt alpha = {ALPHA}:")

DD1_P = delta_D(ALPHA, m1_P2, v1_P2)
DD2_P = delta_D(ALPHA, m2_P2, v2_P2)
DD1_H = delta_D(ALPHA, m1_H, v1_H)
DD2_H = delta_D(ALPHA, m2_H, v2_H)
DDD_P = DD1_P - DD2_P
DDD_H = DD1_H - DD2_H

print(f"  Delta D^P(phi_1)    = {DD1_P:+.4f}")
print(f"  Delta D^P(phi_2)    = {DD2_P:+.4f}")
print(f"  Delta D^H(phi_1)    = {DD1_H:+.4f}")
print(f"  Delta D^H(phi_2)    = {DD2_H:+.4f}")
print(f"  Delta-Delta D^P     = {DDD_P:+.4f}")
print(f"  Delta-Delta D^H     = {DDD_H:+.4f}")

# Lower threshold (dominance preservation)
if DDD_P < 0 and DDD_H > 0:
    eps_lower = -DDD_P / (DDD_H - DDD_P)
else:
    eps_lower = 0.0

# Upper threshold (PRADA preservation)
upper_bounds = []
for i, (DP, DH) in enumerate([(DD1_P, DD1_H), (DD2_P, DD2_H)], start=1):
    if DH < 0:
        upper_bounds.append(DP / (DP - DH))
eps_upper = min(upper_bounds) if upper_bounds else 1.0

print(f"\n  underline_eps   = {eps_lower:.4f}   (dominance preservation)")
print(f"  bar_eps         = {eps_upper:.4f}   (PRADA preservation)")
print(f"  Admissible      = [{eps_lower:.4f}, {eps_upper:.4f}]")

# ============================================================
# EXAMPLE 3: fisher scaling matrix comparison
# ============================================================
"""
BLADE: Comparing Scaling Factors via Delta-Delta-D
====================================================
Model   : Y_t ~ N(0, vartheta),  conditional variance model
Score   : s(y, vartheta) = y^2 / (2*vartheta^2) - 1 / (2*vartheta)
Fisher  : I_{t-1} = 1 / (2*vartheta^2)

Two updates phi_i(y, vartheta) = vartheta + alpha * S^(i) * s(y, vartheta):
  phi_1 : S^(1) = I^{-1} = 2*vartheta^2   (inverse Fisher)
  phi_2 : S^(2) = 1                        (identity)

Evaluated at vartheta_{t|t-1} = 1 under P_t = N(0, 2).

Three objects are computed:
  1. Delta-Delta-D : contraction dominance difference  -> threshold alpha*
  2. Delta-D(phi_1): individual divergence reduction   -> PRADA bound alpha_bar_1
  3. Delta-D(phi_2): individual divergence reduction   -> PRADA bound alpha_bar_2
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# ── Parameters ────────────────────────────────────────────────────────────────
sigma2_true = 2.0    # P_t = N(0, 2)
S1          = 2.0    # inverse Fisher scaling at vartheta=1: S^(1) = 2*1^2 = 2
S2          = 1.0    # identity scaling

# ── Updated variances ─────────────────────────────────────────────────────────
# s(ycheck, 1) = 1/2 * (ycheck^2 - 1)
# phi_i(ycheck, 1) = 1 + alpha * S^(i) * s(ycheck, 1)

def phi1(ycheck, alpha):
    return 1.0 + alpha * S1 * 0.5 * (ycheck**2 - 1)   # = 1 + alpha*(ycheck^2 - 1)

def phi2(ycheck, alpha):
    return 1.0 + alpha * S2 * 0.5 * (ycheck**2 - 1)   # = 1 + (alpha/2)*(ycheck^2 - 1)

# ── Density of P_t = N(0, sigma2_true) ───────────────────────────────────────
def p(ycheck):
    return np.exp(-ycheck**2 / (2 * sigma2_true)) / np.sqrt(2 * np.pi * sigma2_true)

# ── Object 1: Delta-Delta-D ───────────────────────────────────────────────────
# Definition 3 with R_t = P_t, after factoring out the inner expectation over Y_t:
#
#   Delta-Delta-D = E_{P_t x P_t}[LogS(phi_1(Ycheck, 1), Y) - LogS(phi_2(Ycheck, 1), Y)]
#
# Inner expectation over Y_t gives S_bar(vartheta', P_t) = -1/2*log(2*pi*vartheta') - 1/vartheta'.
# Taking the difference between phi_1 and phi_2, the -1/2*log(2*pi) terms cancel, leaving:
#
#   integrand = [ -1/2 * log(phi1/phi2) - 1/phi1 + 1/phi2 ] * p(ycheck)

def integrand_ddD(ycheck, alpha):
    v1 = phi1(ycheck, alpha)
    v2 = phi2(ycheck, alpha)
    return (-0.5 * np.log(v1 / v2) - 1.0 / v1 + 1.0 / v2) * p(ycheck)

def delta_delta_D(alpha):
    val, _ = quad(integrand_ddD, -np.inf, np.inf, args=(alpha,))
    return val

# ── Object 2: Delta-D (individual divergence reduction) ──────────────────────
# Delta-D^{P_t}(phi_i) = E_{P_t x P_t}[LogS(phi_i(Ycheck, 1), Y) - LogS(1, Y)]
#
# Inner expectation gives S_bar(vartheta', P_t) - S_bar(1, P_t).
# With sigma2_true = 2:
#   S_bar(vartheta', P_t) = -1/2*log(2*pi*vartheta') - 1/vartheta'
#   S_bar(1,         P_t) = -1/2*log(2*pi)           - 1
# Difference: -1/2*log(vartheta') - 1/vartheta' + 1
#
#   integrand = [ -1/2 * log(vp) - 1/vp + 1 ] * p(ycheck)

def integrand_dD(ycheck, alpha, S):
    vp = 1.0 + alpha * S * 0.5 * (ycheck**2 - 1)
    return (-0.5 * np.log(vp) - 1.0 / vp + 1.0) * p(ycheck)

def delta_D(alpha, S):
    val, _ = quad(integrand_dD, -np.inf, np.inf, args=(alpha, S))
    return val

# ── Find thresholds ───────────────────────────────────────────────────────────
# Grid capped at alpha < 1: at alpha = 1, phi_1(0, 1) = 0 making log undefined.
# All thresholds of interest (alpha* ~ 0.109, alpha_bar_1 ~ 0.195) are well below 1.

alpha_grid = np.linspace(1e-4, 0.99, 2000)

def find_zero(f, grid):
    vals = np.array([f(a) for a in grid])
    idx  = np.where(np.diff(np.sign(vals)) < 0)[0][0]
    return brentq(f, grid[idx], grid[idx + 1], xtol=1e-10)

alpha_star  = find_zero(delta_delta_D,            alpha_grid)
alpha_bar_1 = find_zero(lambda a: delta_D(a, S1), alpha_grid)
alpha_bar_2 = find_zero(lambda a: delta_D(a, S2), alpha_grid)
alpha_bar   = min(alpha_bar_1, alpha_bar_2)

# ── Results ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("BLADE scaling comparison: N(0, vartheta) variance model")
print("=" * 55)
print(f"  Dominance threshold   alpha*     = {alpha_star:.4f}")
print(f"  PRADA bound phi_1     alpha_bar1 = {alpha_bar_1:.4f}")
print(f"  PRADA bound phi_2     alpha_bar2 = {alpha_bar_2:.4f}")
print(f"  Binding PRADA bound   alpha_bar  = {alpha_bar:.4f}")
print(f"  Ratio alpha_bar2 / alpha_bar1    = {alpha_bar_2 / alpha_bar_1:.4f}")
print()
print(f"  Dominance interval : (0, {alpha_star:.3f})")
print(f"  PRADA interval     : (0, {alpha_bar:.3f})")
print(f"  Inclusion (0,{alpha_star:.3f}) subset (0,{alpha_bar:.3f}): "
      f"{alpha_star < alpha_bar}")