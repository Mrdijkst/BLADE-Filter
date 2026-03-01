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


def student_t_loglik(y, sigma2_hat, nu):
    """
    Full predictive log-likelihood for volatility model:
        y_t | F_{t-1} ~ sqrt(sigma2_t) * t_nu(0,1)
    Returns average log-likelihood.
    """
    std_resid = y / np.sqrt(sigma2_hat)

    c = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(np.pi * nu)
    )

    ll = (
        c
        - 0.5 * np.log(sigma2_hat)              # Jacobian term
        - ((nu + 1) / 2) * np.log(1 + std_resid**2 / nu)
    )

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


def simulate_scale_dgp_obs_outliers(
    T_train,
    T_test,
    omega,
    alpha,
    beta,
    nu,                 # df for CLEAN eta_t (choose large nu for "more normal")
    p_out=0.01,         # outlier probability
    out_scale=10.0,     # outlier size in "sigmas" (see below)
    outlier_dist="t",   # "t" or "normal"
    outlier_df=3,       # df for outlier t (if used)
    contam_where="both",  # "train", "test", "both"
    seed=0,
):
    """
    Clean latent process:
        y_clean[t] = sqrt(sigma2[t]) * eta_t, eta_t ~ t_nu
        sigma2[t+1] = omega + alpha*y_clean[t]^2 + beta*sigma2[t]

    Observational outliers:
        y_obs[t] = y_clean[t] + I_t * out_t
    where out_t has scale out_scale * sqrt(sigma2[t]) (so it's big relative to current vol).
    """
    rng = np.random.default_rng(seed)
    T = T_train + T_test

    y_clean = np.zeros(T)
    y_obs   = np.zeros(T)
    sigma2  = np.zeros(T + 1)

    # unconditional variance init (requires alpha+beta<1)
    sigma2[0] = omega / (1.0 - alpha - beta)

    # clean innovations
    eta = rng.standard_t(nu, size=T)

    # simulate clean latent path first
    for t in range(T):
        y_clean[t] = np.sqrt(sigma2[t]) * eta[t]
        sigma2[t + 1] = omega + alpha * (y_clean[t]**2 - sigma2[t]) + beta * sigma2[t]

    # decide which indices get contaminated
    idx = np.arange(T)
    if contam_where == "train":
        cand = idx[:T_train]
    elif contam_where == "test":
        cand = idx[T_train:]
    elif contam_where == "both":
        cand = idx
    else:
        raise ValueError("contam_where must be 'train', 'test', or 'both'")

    I = np.zeros(T, dtype=bool)
    I[cand] = rng.random(len(cand)) < p_out

    # draw outlier shocks (additive)
    if outlier_dist == "normal":
        z = rng.standard_normal(T)
    elif outlier_dist == "t":
        z = rng.standard_t(outlier_df, size=T)
    else:
        raise ValueError("outlier_dist must be 'normal' or 't'")

    # scale outliers by current conditional sd to make them "k sigmas"
    out = out_scale * np.sqrt(sigma2[:T]) * z

    # observed series
    y_obs = y_clean + I * out

    # return splits + true sigma2 aligned with y[t] uses sigma2[t]
    sigma2_true_train = sigma2[:T_train]
    sigma2_true_test  = sigma2[T_train:T_train + T_test]

    return (
        y_clean[:T_train], y_clean[T_train:],
        y_obs[:T_train],   y_obs[T_train:],
        sigma2_true_train, sigma2_true_test,
        I[:T_train], I[T_train:]
    )


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
    nu_clean = nu

    (
        y_train_obs, y_test_obs, sigma2_true_test,
    ) = simulate_scale_dgp_obs_outliers(
        T_train=T_train,
        T_test=T_test,
        omega=omega0,
        alpha=alpha0,
        beta=beta0,
        nu=nu_clean,
        p_out=1.0 - eps,          # if you want eps = "fraction clean"
        out_scale=10.0,
        outlier_dist="t",
        outlier_df=3,
        contam_where="train",     # Which part of the sample to contaminate: "train", "test", or "both"
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
       # fit models on OBSERVED contaminated train
        model.fit(y_train_obs)

        # predict on OBSERVED test (clean if contam_where="train")
        sigma2_hat = one_step_ahead_scale(model, y_train_obs, y_test_obs)

        # evaluate against TRUE latent sigma2 path
        errors = sigma2_true_test - sigma2_hat

        for i, gamma2 in enumerate(gamma2_list):

            if gamma2 == 1:
                LossMat[i, j] = np.mean(np.abs(errors))

            elif gamma2 == 2:
                LossMat[i, j] = np.mean(errors**2)

            elif gamma2 == "Loglik":
                LossMat[i, j] = student_t_loglik(y_test_obs, sigma2_hat, nu=nu_clean)

            else:
                LossMat[i, j] = np.mean(barron_loss_pos_vec(errors, gamma2, xi=1.0))

    # --------------------------------------------------
    # Oracle benchmark: Beta-t GARCH
    # --------------------------------------------------
    oracle_col = len(gamma1_grid)

    oracle = Beta_t_GARCH11(y_train_obs)
    oracle.fit()
    sigma2_oracle = one_step_ahead_beta_t_garch(oracle, y_train_obs, y_test_obs)
    errors_oracle = sigma2_true_test - sigma2_oracle

    for i, gamma2 in enumerate(gamma2_list):

        if gamma2 == 1:
            LossMat[i, oracle_col] = np.mean(np.abs(errors_oracle))

        elif gamma2 == 2:
            LossMat[i, oracle_col] = np.mean(errors_oracle**2)

        elif gamma2 == "Loglik":
            LossMat[i, oracle_col] = student_t_loglik(y_test_obs, sigma2_oracle, nu=nu_clean)

        else:
            LossMat[i, oracle_col] = np.mean(barron_loss_pos_vec(errors_oracle, gamma2, xi=1.0))

    return LossMat


