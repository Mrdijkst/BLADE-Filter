import numpy as np
from scipy.special import gammaln
from All_models_for_montecarlo import RobustQLEModel, Beta_t_GARCH11


# ============================================================
# Losses / utilities
# ============================================================

def barron_loss_pos_vec(e, gamma, xi=1.0):
    z2 = (e / xi) ** 2

    if gamma == 2:
        return 0.5 * z2
    if gamma == 0:
        return np.log1p(0.5 * z2)
    if gamma == -np.inf:
        return 1.0 - np.exp(-0.5 * z2)

    return (np.abs(gamma - 2.0) / gamma) * (
        (1.0 + z2 / (np.abs(gamma - 2.0))) ** (gamma / 2.0) - 1.0
    )


def student_t_loglik(std_resid, nu):
    """
    Log-likelihood of Student-t innovations with df=nu, mean 0, scale 1.
    Returns the *average* log-likelihood.
    """
    c = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(np.pi * nu)
    )
    ll = c - ((nu + 1) / 2) * np.log(1 + (std_resid**2) / nu)
    return np.mean(ll)


def nu_tilde_for_10x_sd(nu):
    """
    Choose nu_tilde so that Var(t_nu_tilde) = 100 * Var(t_nu),
    i.e. SD is 10x larger.
    Var(t_nu) = nu/(nu-2) for nu>2.
    """
    R = 100.0 * (nu / (nu - 2.0))
    return 2.0 * R / (R - 1.0)


# ============================================================
# One-step-ahead prediction: RobustQLE volatility model
# ============================================================

def one_step_ahead_scale(model, y_train, y_test):
    """
    Your RobustQLEModel scale recursion:
        f_{t+1} = omega + gamma * psi(e_t) + beta f_t
    with e_t = y_t^2 - f_t
    """
    
    omega = model.params['omega']
    gamma = model.params['gamma']
    beta  = model.params['beta']
    #print("Model params:", omega, gamma, beta)
    print("Model params:", model.params)

    if model.alpha_loss is None:
        alpha_loss = model.params['alpha_loss']
    else:
        alpha_loss = model.alpha_loss

    if model.c is None:
        c = model.params['c']
    else:
        c = model.c

    f = omega / (1.0 - beta)

    # filter train
    for yt in y_train:
        e = yt**2 - f
        psi = model._rho_derivative(e, alpha_loss, c)
        f = omega + gamma * psi + beta * f

    # one-step-ahead preds on test
    f_pred = np.zeros(len(y_test))
    for t, yt in enumerate(y_test):
        f_pred[t] = f
        e = yt**2 - f
        psi = model._rho_derivative(e, alpha_loss, c)
        f = omega + gamma * psi + beta * f

    return f_pred


# ============================================================
# One-step-ahead prediction: Beta-t GARCH(1,1) oracle
# ============================================================

def one_step_ahead_beta_t_garch(model, y_train, y_test):
    """
    Beta-t GARCH recursion in your Beta_t_GARCH11:
        eps_t = y_t / sqrt(f_t)
        score = (nu+1)*eps_t^2 / (nu-2 + eps_t^2)
        f_{t+1} = omega + alpha*score*f_t + beta*f_t
    """
    if model.params is None:
        raise ValueError("Model must be fitted first")

    omega, alpha, beta, nu = model.params

    # stationary init
    f = omega / (1.0 - alpha - beta)

    # filter train
    for yt in y_train:
        eps = yt / np.sqrt(f)
        score = (nu + 1.0) * eps**2 / (nu - 2.0 + eps**2)
        f = omega + alpha * score * f + beta * f

    # predict test
    f_pred = np.zeros(len(y_test))
    for t, yt in enumerate(y_test):
        f_pred[t] = f
        eps = yt / np.sqrt(f)
        score = (nu + 1.0) * eps**2 / (nu - 2.0 + eps**2)
        f = omega + alpha * score * f + beta * f

    return f_pred


# ============================================================
# DGP: scale model with train-only contamination (mixture of t)
# ============================================================

def simulate_scale_dgp_train_only_heavytail(
    T_train,
    T_test,
    omega,
    alpha,
    beta,
    nu,
    eps,     # contamination mixing prob for CLEAN component in TRAIN (like your code)
    seed
):
    """
    Volatility DGP (GARCH-style state) + Student-t innovations.

    - Train eps_t ~ mixture: eps * t_nu + (1-eps) * t_nu_tilde (heavier tail / larger scale)
    - Test  eps_t ~ t_nu (clean)
    - Vol recursion uses y_t^2 as in your clean scale DGP:
          sigma2_{t+1} = omega + alpha * y_t^2 + beta * sigma2_t

    Returns:
        y_train, y_test, sigma2_true_test  (sigma2_true_test aligned with y_test)
    """
    rng = np.random.default_rng(seed)

    T = T_train + T_test
    y = np.zeros(T)
    sigma2 = np.zeros(T + 1)

    # init at unconditional variance (requires alpha+beta<1)
    sigma2[0] = omega / (1.0 - alpha - beta)

    nu_tilde = nu_tilde_for_10x_sd(nu)

    # TRAIN innovations: symmetric mixture
    u = rng.random(T_train)
    eta_clean = rng.standard_t(nu, size=T_train)
    eta_heavy = rng.standard_t(nu_tilde, size=T_train)
    eta_train = np.where(u < eps, eta_clean, eta_heavy)

    # TEST innovations: clean
    eta_test = rng.standard_t(nu, size=T_test)

    eta = np.concatenate([eta_train, eta_test])

    # simulate
    for t in range(T):
        y[t] = np.sqrt(sigma2[t]) * eta[t]
        sigma2[t + 1] = omega + alpha * (y[t]**2 - sigma2[t]) + beta * sigma2[t]

    # true sigma2_t for test observations y_test corresponds to sigma2[T_train : T_train+T_test]
    sigma2_true_test = sigma2[T_train : T_train + T_test]
    sigma2_true_train = sigma2[:T_train]

    return y[:T_train], y[T_train:], sigma2_true_train, sigma2_true_test


# ============================================================
# One simulation -> loss matrix (TRUE VOL path error)
# ============================================================

def run_one_simulation_scale_train_contaminated(
    nu,
    eps,
    gamma1_grid,
    T_train,
    T_test,
    seed=123
):
    """
    Scale (volatility) Monte Carlo with TRAIN contamination and CLEAN test.
    Losses are computed on out-of-sample *true volatility path*:
        error_t = sigma2_true_test[t] - sigma2_hat[t]
    """

    # --- True (clean) GARCH parameters ---
    omega0 = 0.02
    alpha0 = 0.11
    beta0  = 0.88

    # --- simulate ---
    y_train, y_test, sigma2_true_train ,sigma2_true_test = simulate_scale_dgp_train_only_heavytail(
        T_train=T_train,
        T_test=T_test,
        omega=omega0,
        alpha=alpha0,
        beta=beta0,
        nu=nu,
        eps=eps,
        seed=seed
    )


    gamma2_list = [1, 1.5, 2, 4, "Loglik"]

    n_rows = len(gamma2_list)
    n_cols = len(gamma1_grid) + 1  # + oracle
    LossMat = np.zeros((n_rows, n_cols))

    # --------------------------------------------------
    # RobustQLE volatility models
    # --------------------------------------------------
    for j, gamma1 in enumerate(gamma1_grid):

        model = RobustQLEModel(
            model_type="volatility",
            alpha_loss=gamma1,
            c=1
        )
        model.fit(y_train)

        sigma2_hat = one_step_ahead_scale(model, y_train, y_test)

        # TRUE volatility path error
        errors = sigma2_true_test - sigma2_hat

        for i, gamma2 in enumerate(gamma2_list):

            if gamma2 == 1:
                LossMat[i, j] = np.mean(np.abs(errors))

            elif gamma2 == 2:
                LossMat[i, j] = np.mean(errors**2)

            elif gamma2 == "Loglik":
                # evaluate predictive standardized residuals under CLEAN test nu
                std_resid = y_test / np.sqrt(sigma2_hat)
                LossMat[i, j] = student_t_loglik(std_resid, nu=nu)

            else:
                LossMat[i, j] = np.mean(barron_loss_pos_vec(errors, gamma2, xi=1.0))

    # --------------------------------------------------
    # Oracle benchmark: Beta-t GARCH
    # --------------------------------------------------
    oracle_col = len(gamma1_grid)

    oracle = Beta_t_GARCH11(y_train)
    oracle.fit()

    sigma2_oracle = one_step_ahead_beta_t_garch(oracle, y_train, y_test)

    errors_oracle = sigma2_true_test - sigma2_oracle

    for i, gamma2 in enumerate(gamma2_list):

        if gamma2 == 1:
            LossMat[i, oracle_col] = np.mean(np.abs(errors_oracle))

        elif gamma2 == 2:
            LossMat[i, oracle_col] = np.mean(errors_oracle**2)

        elif gamma2 == "Loglik":
            std_resid = y_test / np.sqrt(sigma2_oracle)
            LossMat[i, oracle_col] = student_t_loglik(std_resid, nu=nu)

        else:
            LossMat[i, oracle_col] = np.mean(barron_loss_pos_vec(errors_oracle, gamma2, xi=1.0))

    return LossMat


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    gamma1_grid = np.array([-30,-20,-10,-3,-1,0,0.5,1,1.5,1.8,2])
    nu = 5
    eps = 0.97

    T_train = 2500
    T_test = 1000
    print(T_train)
    LossMat = run_one_simulation_scale_train_contaminated(
        nu=nu,
        eps=eps,
        gamma1_grid=gamma1_grid,
        T_train=T_train,
        T_test=T_test,
        seed=50
    )

    print("MAE:",        LossMat[0, :])
    print("Barron 1.5:", LossMat[1, :])
    print("MSE:",        LossMat[2, :])
    print("Barron 4:",   LossMat[3, :])
    print("LogLik:",     LossMat[4, :])
