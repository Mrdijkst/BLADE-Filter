import numpy as np
from All_models_for_montecarlo import RobustQLEModel, OracleStudentTLocation
from scipy.special import gammaln
import numpy as np

def barron_loss_pos_vec(e, gamma, xi=1.0):
    z2 = (e / xi) ** 2

    if gamma == 2:
        return 0.5 * z2
    if gamma == 0:
        return np.log1p(0.5 * z2)
    if gamma == -np.inf:
        return 1.0 - np.exp(-0.5 * z2)

    return (np.abs(gamma - 2.0) / gamma) * (
        (1.0 + z2 / (xi**2 * np.abs(gamma - 2.0))) ** (gamma / 2.0) - 1.0
    )

def draw_positive_student_t(nu, size, rng):
    """
    Draw from Student-t(nu) conditional on being positive.
    Density: 2 * t_nu(x) * 1{x > 0}
    """
    out = np.empty(size)
    i = 0
    while i < size:
        draws = rng.standard_t(nu, size=size - i)
        draws = draws[draws > 0]
        if draws.size == 0:
            continue
        take = min(draws.size, size - i)
        out[i:i+take] = draws[:take]
        i += take
    return out

def student_t_loglik(errors, nu):
    """
    Log-likelihood of Student-t innovations with df=nu, mean 0, scale 1.
    Returns the *average* log-likelihood.
    """
    c = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(np.pi * nu)
    )
    ll = c - ((nu + 1) / 2) * np.log(1 + (errors**2) / nu)
    return np.mean(ll)


# ============================================================
# 1. DGP: location model (NO contamination, eps = 1)
# ============================================================

def simulate_location_dgp(
    T_train,
    T_test,
    omega,
    alpha,
    beta,
    nu,
    seed
):
    rng = np.random.default_rng(seed)

    T = T_train + T_test
    y = np.zeros(T)
    theta = np.zeros(T + 1)

    theta[0] = omega / (1 - beta)

    eps = rng.standard_t(nu, size=T)

    for t in range(T):
        y[t] = theta[t] + eps[t]
        theta[t + 1] = omega + alpha * (y[t] - theta[t]) + beta * theta[t]

    return y[:T_train], y[T_train:], theta[T_train:-1]


# ============================================================
# 2. One-step-ahead predictions (PRE-UPDATE!)
# ============================================================

def one_step_ahead_location(model, y_train, y_test):
    params = np.array([model.params[k] for k in model.param_names])

    omega, gamma, beta, c= params[:]
    alpha_loss = model.alpha_loss
    print("Model parameters:", params)
    print("alpha_loss:", alpha_loss, "c:", c)

    # --- filter training sample to get final state ---
    f = omega / (1 - beta)
    for yt in y_train:
        e = yt - f
        psi = model._rho_derivative(e, alpha_loss, c)
        f = omega + gamma * psi + beta * f

    # --- one-step-ahead predictions on test ---
    f_pred = np.zeros(len(y_test))
    for t, yt in enumerate(y_test):
        f_pred[t] = f
        e = yt - f
        psi = model._rho_derivative(e, alpha_loss, c)
        f = omega + gamma * psi + beta * f

    return f_pred


# ============================================================
# 3. ONE SIMULATION → ONE WHITEBOARD MATRIX
# ============================================================

def run_one_simulation_upper_left(
    nu,
    gamma1_grid,
    T_train=2500,
    T_test=1000,
    seed=123
):
    """
    Upper-left panel:
    - eps = 1 (no contamination)
    - non-distorted location
    - nu fixed in DGP
    - gamma1 varies in estimation
    - oracle Student-t benchmark added as extra column
    """

    # --------------------------------------------------
    # 1. True DGP parameters
    # --------------------------------------------------
    omega0 = 0.0
    alpha0 = 0.1
    beta0  = 0.3

    # --------------------------------------------------
    # 2. Simulate data
    # --------------------------------------------------
    y_train, y_test, theta_test = simulate_location_dgp(
        T_train, T_test,
        omega0, alpha0, beta0,
        nu,
        seed
    )

    # --------------------------------------------------
    # 3. Loss setup (rows)
    # --------------------------------------------------
    gamma2_list = [1, 1.5, 2, 4, "Loglik"]   # MAE, Barron(1.5), MSE, Barron(4)

    # --------------------------------------------------
    # 4. Allocate whiteboard matrix
    #    columns = gamma1 models + oracle
    # --------------------------------------------------
    n_benchmarks = 1
    n_rows = len(gamma2_list)
    n_cols = len(gamma1_grid) + n_benchmarks

    LossMat = np.zeros((n_rows, n_cols))

    # --------------------------------------------------
    # 5. Loop over gamma1 models
    # --------------------------------------------------
    for j, gamma1 in enumerate(gamma1_grid):

        model = RobustQLEModel(
            model_type="location",
            alpha_loss=gamma1
        )

        model.fit(y_train)


        y_hat = one_step_ahead_location(model, y_train, y_test)
        errors = y_test - y_hat

        for i, gamma2 in enumerate(gamma2_list):

            if gamma2 == 1:
                LossMat[i, j] = np.mean(np.abs(errors))        # MAE

            elif gamma2 == 2:
                LossMat[i, j] = np.mean(errors**2)             # MSE

            elif gamma2 == "Loglik":
                LossMat[i, j] = student_t_loglik(errors, nu=5)  # pseudo-LL

            else:
                LossMat[i, j] = np.mean(
                    barron_loss_pos_vec(errors, gamma2, xi=1.0)
                )

    # --------------------------------------------------
    # 6. Oracle Student-t benchmark (run ONCE)
    # --------------------------------------------------
    oracle_col = len(gamma1_grid)

    oracle = OracleStudentTLocation(y_train, nu=nu)
    oracle.fit()

    y_hat_oracle = oracle.one_step_ahead(y_test)
    errors_oracle = y_test - y_hat_oracle

    for i, gamma2 in enumerate(gamma2_list):

        if gamma2 == 1:
            LossMat[i, oracle_col] = np.mean(np.abs(errors_oracle))

        elif gamma2 == 2:
            LossMat[i, oracle_col] = np.mean(errors_oracle**2)

        elif gamma2 == "Loglik":
            # TRUE likelihood for oracle
            LossMat[i, oracle_col] = student_t_loglik(errors_oracle, nu=nu)

        else:
            LossMat[i, oracle_col] = np.mean(
                barron_loss_pos_vec(errors_oracle, gamma2, xi=1.0)
            )

    return LossMat

# ============================================================
# 4. Example usage
# ============================================================

if __name__ == "__main__":

    gamma1_grid = np.array([-5,-3,-2,-1,-0.5, 0, 0.5, 1.0, 2])  # example gamma1 values
    nu = 3  # one ν-panel value

    LossMat = run_one_simulation_upper_left(
        nu=nu,
        gamma1_grid=gamma1_grid,
        seed=43
    )

    print("MAE:",        LossMat[0, :])
    print("Barron 1.5:", LossMat[1, :])
    print("MSE:",        LossMat[2, :])
    print("Barron 4:",   LossMat[3, :])
    print("LogLik:",     LossMat[4, :])