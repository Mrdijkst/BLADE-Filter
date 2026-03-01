import numpy as np
from All_models_for_montecarlo import RobustQLEModel, OracleStudentTLocation, Beta_t_GARCH11
from scipy.special import gammaln


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

def one_step_ahead_beta_t_garch(model, y_train, y_test):
    """
    One-step-ahead volatility forecasts for Beta-t GARCH(1,1).
    
    Returns:
        f_pred : array of predicted conditional variances
                 E_t[f_{t+1}]
    """

    if model.params is None:
        raise ValueError("Model must be fitted first")

    omega, alpha, beta, nu = model.params

    print("Model parameters:", model.params)

    # --- Initialize variance ---
    # Use unconditional mean (stationary case)
    f = omega / (1 - alpha - beta)

    # --- Filter through training sample ---
    for yt in y_train:
        eps = yt / np.sqrt(f)
        score = (nu + 1) * eps**2 / (nu - 2 + eps**2)
        f = omega + alpha * score * f + beta * f

    # --- One-step-ahead predictions ---
    f_pred = np.zeros(len(y_test))

    for t, yt in enumerate(y_test):

        # forecast available before seeing y_t
        f_pred[t] = f

        # update using realized y_t
        eps = yt / np.sqrt(f)
        score = (nu + 1) * eps**2 / (nu - 2 + eps**2)
        f = omega + alpha * score * f + beta * f

    return f_pred

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

# ============================================================
# 1. DGP: Student-t GARCH(1,1) scale model
# ============================================================

def simulate_scale_dgp(
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
    sigma2 = np.zeros(T + 1)
    

    # unconditional variance
    sigma2[0] = omega / (1 - alpha - beta)

    eps = rng.standard_t(nu, size=T)

    for t in range(T):
        y[t] = np.sqrt(sigma2[t]) * eps[t]
        sigma2[t + 1] = omega + alpha * y[t]**2 + beta * sigma2[t]

    return y[:T_train], y[T_train:]

# ============================================================
# 2. One-step-ahead volatility predictions
# ============================================================

def one_step_ahead_scale(model, y_train, y_test):
    params = np.array([model.params[k] for k in model.param_names])

    omega, gamma, beta = params[:3]
    alpha_loss = model.alpha_loss
    c = model.c
    #pritn model parameters with names

    print("Model parameters:", params)
    print("alpha_loss:", alpha_loss, "c:", c)

    # initialize variance
    f = omega / (1 - beta)

    # --- filter training ---
    for yt in y_train:
        e = yt**2 -f
        psi = model._rho_derivative(e, alpha_loss, c)
        f = omega + gamma * psi + beta * f

    # --- predict on test ---
    f_pred = np.zeros(len(y_test))

    for t, yt in enumerate(y_test):
        f_pred[t] = f
        e = yt**2 - f
        psi = model._rho_derivative(e, alpha_loss, c)
        f = omega + gamma * psi + beta * f

    return f_pred


# ============================================================
# 3. ONE SIMULATION → VOLATILITY MATRIX
# ============================================================

def run_one_simulation_scale(
    nu,
    gamma1_grid,
    T_train,
    T_test,
    seed=123
):
    """
    Scale (volatility) Monte Carlo
    """

    # True GARCH parameters
    omega0 = 0.02
    alpha0 = 0.11
    beta0  = 0.72

    # Simulate
    y_train, y_test = simulate_scale_dgp(
        T_train, T_test,
        omega0, alpha0, beta0,
        nu,
        seed
    )

    gamma2_list = [1, 1.5, 2, 4, "Loglik"]

    n_rows = len(gamma2_list)
    n_cols = len(gamma1_grid) + 1  # + oracle
    LossMat = np.zeros((n_rows, n_cols))

    # --------------------------------------------------
    # Estimate Barron volatility models
    # --------------------------------------------------
    for j, gamma1 in enumerate(gamma1_grid):

        model = RobustQLEModel(
            model_type="volatility",
            alpha_loss=gamma1,
            c=1
        )
        model.fit(y_train)

        sigma2_hat = one_step_ahead_scale(model, y_train, y_test)
        errors = y_test**2 - sigma2_hat  # volatility error

        for i, gamma2 in enumerate(gamma2_list):

            if gamma2 == 1:
                LossMat[i, j] = np.mean(np.abs(errors))

            elif gamma2 == 2:
                LossMat[i, j] = np.mean(errors**2)

            elif gamma2 == "Loglik":
                std_resid = y_test / np.sqrt(sigma2_hat)
                LossMat[i, j] = student_t_loglik(std_resid, nu=nu)

            else:
                LossMat[i, j] = np.mean(
                    barron_loss_pos_vec(errors, gamma2, xi=1.0)
                )

    # --------------------------------------------------
    # Oracle Student-t GARCH benchmark
    # --------------------------------------------------
    oracle_col = len(gamma1_grid)

    oracle = Beta_t_GARCH11(y_train)
    oracle.fit()

    sigma2_oracle = one_step_ahead_beta_t_garch(oracle, y_train, y_test)
    errors_oracle = y_test**2 - sigma2_oracle

    for i, gamma2 in enumerate(gamma2_list):

        if gamma2 == 1:
            LossMat[i, oracle_col] = np.mean(np.abs(errors_oracle))

        elif gamma2 == 2:
            LossMat[i, oracle_col] = np.mean(errors_oracle**2)

        elif gamma2 == "Loglik":
            std_resid = y_test / np.sqrt(sigma2_oracle)
            LossMat[i, oracle_col] = student_t_loglik(std_resid, nu=nu)

        else:
            LossMat[i, oracle_col] = np.mean(
                barron_loss_pos_vec(errors_oracle, gamma2, xi=1.0)
            )

    return LossMat


# ============================================================
# 4. Example usage
# ============================================================

if __name__ == "__main__":

    gamma1_grid = np.array([-1,0,0.5,1,1.5,2])
    nu = 8
    
    T_train = 2500
    T_test = 1000

    LossMat = run_one_simulation_scale(
        nu=nu,
        gamma1_grid=gamma1_grid,
        T_train=T_train,
        T_test = T_test,
        seed=43
    )

    print("MAE:",        LossMat[0, :])
    print("Barron 1.5:", LossMat[1, :])
    print("MSE:",        LossMat[2, :])
    print("Barron 4:",   LossMat[3, :])
    print("LogLik:",     LossMat[4, :])
