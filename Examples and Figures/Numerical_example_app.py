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