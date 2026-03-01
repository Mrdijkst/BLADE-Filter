import numpy as np
from All_models_for_montecarlo import RobustQLEModel, OracleStudentTLocation, Beta_t_GARCH11
from scipy.special import gammaln


# ============================================================
# Barron loss
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


# ============================================================
# Student-t loglik
# ============================================================

def student_t_loglik(errors, nu):
    c = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(np.pi * nu)
    )
    ll = c - ((nu + 1) / 2) * np.log(1 + (errors**2) / nu)
    return np.mean(ll)


# ============================================================
# DGP: Student-t GARCH(1,1)
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

    sigma2[0] = omega / (1 - alpha - beta)

    eps = rng.standard_t(nu, size=T)

    for t in range(T):
        y[t] = np.sqrt(sigma2[t]) * eps[t]
        sigma2[t + 1] = omega + alpha * (y[t]**2 - sigma2[t]) + beta * sigma2[t]

    # return TRUE volatility path for test sample
    return (
        y[:T_train],
        y[T_train:],
        sigma2[T_train:T_train + T_test]   # true sigma_t^2 for test
    )


# ============================================================
# One-step-ahead forecasts
# ============================================================

def one_step_ahead_scale(model, y_train, y_test):

    params = np.array([model.params[k] for k in model.param_names])
    omega, gamma, beta = params[:3]
    alpha_loss = model.alpha_loss
    c = model.c

    f = omega / (1 - beta)

    # filter training
    for yt in y_train:
        e = yt**2 - f
        psi = model._rho_derivative(e, alpha_loss, c)
        f = omega + gamma * psi + beta * f

    # predict test
    f_pred = np.zeros(len(y_test))

    for t, yt in enumerate(y_test):
        f_pred[t] = f
        e = yt**2 - f
        psi = model._rho_derivative(e, alpha_loss, c)
        f = omega + gamma * psi + beta * f

    return f_pred


def one_step_ahead_beta_t_garch(model, y_train, y_test):

    omega, alpha, beta, nu = model.params

    f = omega / (1 - alpha - beta)

    for yt in y_train:
        eps = yt / np.sqrt(f)
        score = (nu + 1) * eps**2 / (nu - 2 + eps**2)
        f = omega + alpha * score * f + beta * f

    f_pred = np.zeros(len(y_test))

    for t, yt in enumerate(y_test):
        f_pred[t] = f
        eps = yt / np.sqrt(f)
        score = (nu + 1) * eps**2 / (nu - 2 + eps**2)
        f = omega + alpha * score * f + beta * f

    return f_pred


# ============================================================
# Monte Carlo simulation
# ============================================================

def run_one_simulation_scale(
    nu,
    gamma1_grid,
    T_train,
    T_test,
    seed=123
):

    omega0 = 0.02
    alpha0 = 0.11
    beta0  = 0.72

    y_train, y_test, sigma2_true_test = simulate_scale_dgp(
        T_train, T_test,
        omega0, alpha0, beta0,
        nu,
        seed
    )

    gamma2_list = [1, 1.5, 2, 4, "Loglik"]

    n_rows = len(gamma2_list)
    n_cols = len(gamma1_grid) + 1
    LossMat = np.zeros((n_rows, n_cols))

    # --------------------------------------------------
    # Barron volatility models
    # --------------------------------------------------
    for j, gamma1 in enumerate(gamma1_grid):

        model = RobustQLEModel(
            model_type="volatility",
            alpha_loss=gamma1,
            c=1
        )
        model.fit(y_train)

        sigma2_hat = one_step_ahead_scale(model, y_train, y_test)

        # TRUE volatility error
        errors = sigma2_true_test - sigma2_hat

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
                    barron_loss_pos_vec(errors, gamma2)
                )

    # --------------------------------------------------
    # Oracle Student-t GARCH
    # --------------------------------------------------
    oracle_col = len(gamma1_grid)

    oracle = Beta_t_GARCH11(y_train)
    oracle.fit()

    sigma2_oracle = one_step_ahead_beta_t_garch(
        oracle, y_train, y_test
    )

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
            LossMat[i, oracle_col] = np.mean(
                barron_loss_pos_vec(errors_oracle, gamma2)
            )

    return LossMat


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    gamma1_grid = np.array([-5,-1, 0, 0.5, 1, 1.5, 2])
    nu = 3


    T_train = 4000
    T_test = 1000

    LossMat = run_one_simulation_scale(
        nu=nu,
        gamma1_grid=gamma1_grid,
        T_train=T_train,
        T_test=T_test,
        seed=43
    )

    print("MAE:",        LossMat[0, :])
    print("Barron 1.5:", LossMat[1, :])
    print("MSE:",        LossMat[2, :])
    print("Barron 4:",   LossMat[3, :])
    print("LogLik:",     LossMat[4, :])
