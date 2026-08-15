import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit, gammaln, logsumexp

import warnings
warnings.filterwarnings('ignore')
    


class GARCH11:
    """
    Purpose
    -------
    Estimate a Gaussian GARCH(1,1) model by maximum likelihood.

    Model
    -----
    y_t = sqrt(h_t) * epsilon_t

    epsilon_t ~ N(0,1)

    h_t = omega + alpha * y_{t-1}^2 + beta * h_{t-1}

    Parameters
    ----------
    vY : ndarray
        Return series.

    Attributes
    ----------
    params : tuple
        Estimated parameters (omega, alpha, beta).

    fitted_f : ndarray
        Filtered conditional variance sequence.
    """

    def __init__(self, vY):

        self.vY = np.asarray(vY)

        self.params = None
        self.fitted_f = None

    def _filter(self, vParams):
        """
        Compute the conditional variance recursion.

        Parameters
        ----------
        vParams : array_like
            Parameter vector (omega, alpha, beta).

        Returns
        -------
        ndarray
            Conditional variance path.
        """

        dOmega, dAlpha, dBeta = vParams

        iT = len(self.vY)

        vH = np.zeros(iT)

        vH[0] = dOmega / max(1.0 - dAlpha - dBeta, 1e-6)

        for t in range(1, iT):

            vH[t] = (
                dOmega
                + dAlpha * self.vY[t - 1]**2
                + dBeta * vH[t - 1]
            )

        return vH
    
    def _link(self, vParams):

        omega, alpha, beta = vParams

        a = alpha
        b = beta / (1.0 - alpha)

        return np.array([
            np.log(omega),
            logit(a),
            logit(b)
        ])


    def _link_inverse(self, vParamsTr):

        zOmega, zAlpha, zBeta = vParamsTr

        omega = np.exp(zOmega)

        a = expit(zAlpha)
        b = expit(zBeta)

        alpha = a
        beta = (1.0 - alpha) * b

        return np.array([omega, alpha, beta])
    
    def _nll(self, vParams):

        vH = self._filter(vParams)

        if np.any(vH <= 0) or np.any(~np.isfinite(vH)):
            return 1e10

        return np.mean(
            0.5 * (
                np.log(vH)
                + self.vY**2 / vH
            )
        )
        
    def _nll_tr(self, vParamsTr):

        vParams = self._link_inverse(vParamsTr)

        return self._nll(vParams)

    def fit(self):

        vStart = np.array([
            0.2,
            0.1,
            0.80
        ])

        vStartTr = self._link(vStart)

        oRes = minimize(
            self._nll_tr,
            vStartTr,
            method="L-BFGS-B"
        )

        vParams = self._link_inverse(oRes.x)

        self.params = tuple(vParams)
        self.fitted_f = self._filter(vParams)

        return self
    
class Student_t_GARCH11:
    """
    Purpose
    -------
    Estimate a Student-t GARCH(1,1) model by maximum likelihood.

    Model
    -----
    y_t = sqrt(h_t) * epsilon_t,   epsilon_t ~ t_nu  (nu > 2, unit variance)

    h_t = omega + alpha * y_{t-1}^2 + beta * h_{t-1}

    Parameters
    ----------
    vY : ndarray
        Return series.

    Attributes
    ----------
    params : tuple
        Estimated parameters (omega, alpha, beta, nu).

    fitted_f : ndarray
        Filtered conditional variance sequence.
    """

    def __init__(self, vY):

        self.vY = np.asarray(vY, dtype=float)

        self.params = None
        self.fitted_f = None

    def _filter(self, vParams):
        """
        Compute the conditional variance recursion.

        Parameters
        ----------
        vParams : array_like
            Parameter vector (omega, alpha, beta, nu); nu does not enter the
            variance recursion, only the likelihood.

        Returns
        -------
        ndarray
            Conditional variance path.
        """

        dOmega, dAlpha, dBeta, dNu = vParams

        iT = len(self.vY)

        vH = np.zeros(iT)

        vH[0] = dOmega / max(1.0 - dAlpha - dBeta, 1e-6)

        for t in range(1, iT):

            vH[t] = (
                dOmega
                + dAlpha * self.vY[t - 1]**2
                + dBeta * vH[t - 1]
            )

        return vH

    def _link(self, vParams):

        omega, alpha, beta, nu = vParams

        a = alpha
        b = beta / (1.0 - alpha)

        return np.array([
            np.log(omega),
            logit(a),
            logit(b),
            np.log(nu - 2.0)
        ])

    def _link_inverse(self, vParamsTr):

        zOmega, zAlpha, zBeta, zNu = vParamsTr

        omega = np.exp(zOmega)

        a = expit(zAlpha)
        b = expit(zBeta)

        alpha = a
        beta = (1.0 - alpha) * b

        nu = 2.0 + np.exp(zNu)

        return np.array([omega, alpha, beta, nu])

    def _nll(self, vParams):

        dOmega, dAlpha, dBeta, dNu = vParams

        vH = self._filter(vParams)

        if np.any(vH <= 0) or np.any(~np.isfinite(vH)):
            return 1e10

        dConst = (
            gammaln((dNu + 1.0) / 2.0)
            - gammaln(dNu / 2.0)
            - 0.5 * np.log(np.pi * (dNu - 2.0))
        )

        vLL = (
            dConst
            - 0.5 * np.log(vH)
            - 0.5 * (dNu + 1.0)
              * np.log(1.0 + self.vY**2 / ((dNu - 2.0) * vH))
        )

        return -np.mean(vLL)

    def _nll_tr(self, vParamsTr):

        vParams = self._link_inverse(vParamsTr)

        return self._nll(vParams)

    def fit(self):

        vStart = np.array([
            0.2,
            0.1,
            0.80,
            6.0
        ])

        vStartTr = self._link(vStart)

        oRes = minimize(
            self._nll_tr,
            vStartTr,
            method="L-BFGS-B"
        )

        vParams = self._link_inverse(oRes.x)

        self.params = tuple(vParams)
        self.fitted_f = self._filter(vParams)

        return self

    def get_fitted_variances(self):

        return self.fitted_f

    def forecast_variance(self, horizon=1):

        dOmega, dAlpha, dBeta, dNu = self.params

        dPersistence = dAlpha + dBeta

        dHT = self.fitted_f[-1]
        dYT = self.vY[-1]

        vForecasts = np.zeros(horizon)

        # h = 1 is exact: h_{T+1} = omega + alpha y_T^2 + beta h_T
        dH1 = dOmega + dAlpha * dYT**2 + dBeta * dHT
        vForecasts[0] = dH1

        # h > 1: E[y_{T+j}^2] = h_{T+j}, so variance mean-reverts geometrically
        dUncond = dOmega / (1.0 - dPersistence) if dPersistence < 1.0 else dH1

        for h in range(1, horizon):
            vForecasts[h] = dUncond + dPersistence**h * (dH1 - dUncond)

        return vForecasts

class Beta_t_GARCH11:
    def __init__(self, y):
        """
        β_t GARCH(1,1) with Student‐t innovations:
        
        y_t = sqrt(f_t) * ε_t,   ε_t ~ t_ν   (ν > 2)
        
        Recursion:
             ε_t = y_t / sqrt(f_t)
             f_{t+1} = ω 
                        + α * [ (ν + 1) * ε_t^2 / (ν - 2 + ε_t^2 ) ] 
                        + β * f_t
        
        The log‐likelihood at time t (for y_t | f_t) is that of a 
        Student‐t with ν degrees of freedom, zero mean, and scale = sqrt(f_t).
        """
        self.y = np.asarray(y, dtype=float)
        self.T = len(self.y)

    def _link(self, params):

        omega, alpha, beta, nu = params

        a = alpha
        b = beta / (1.0 - alpha)

        return np.array([
            np.log(omega),
            logit(a),
            logit(b),
            np.log(nu - 2.0)
        ])
 
    def _link_inverse(self, params_tr):

        z_omega, z_alpha, z_beta, z_nu = params_tr

        omega = np.exp(z_omega)

        a = expit(z_alpha)
        b = expit(z_beta)

        alpha = a
        beta = (1.0 - alpha) * b

        nu = 2.0 + np.exp(z_nu)

        return np.array([
            omega,
            alpha,
            beta,
            nu
        ])
    def _nll_tr(self, params_tr):

        params = self._link_inverse(params_tr)

        nll, _ = self._compute_nll_and_variances(params)

        return nll
    
    def _compute_nll_and_variances(self, params):
        """
        Given params = [omega, alpha, beta], construct {f_t}_{t=0..T-1} in‐sample 
        and return (negative log‐likelihood, the array f).
        """
        ω, α, β, ν = params

        
        # Container for f_t
        f = np.zeros(self.T)
        nll = 0.0
        
        # Initialize f[0] at the sample variance of y (to avoid zero)
        sample_var = np.var(self.y)
        f[0] = sample_var + 1e-8
        
        # Precompute constants for the Student‐t density:
        #   const_part = Γ((ν+1)/2) - Γ(ν/2) - 0.5*log[π (ν - 2)]
        const_part = (
            gammaln( (ν + 1.0) / 2.0 ) 
            - gammaln( ν / 2.0 ) 
            - 0.5 * np.log( np.pi * (ν - 2.0) )
        )
        
        for t in range(self.T - 1):
            yt = self.y[t]
            ft = f[t]
            
            # Standardized residual ε_t
            eps_t = yt / np.sqrt(ft)
            
            # Student‐t log‐density at time t:
            #   log p(y_t | f_t) 
            #   = const_part 
            #     - 0.5 * log(f_t) 
            #     - ((ν + 1)/2) * log[ 1 + (y_t^2) / ((ν - 2) f_t) ].
            logpdf_t = (
                const_part 
                - 0.5 * np.log(ft)
                - ((ν + 1.0) / 2.0) 
                  * np.log( 1.0 + (yt * yt) / ((ν - 2.0) * ft) )
            )


            nll -= logpdf_t  # accumulate negative log‐likelihood
            
            # Now update f[t+1]:
            #   f[t+1] = ω + α * [ (ν + 1) * ε_t^2 / (ν - 2 + ε_t^2) ] + β * f_t
            numerator   = (ν + 1.0) * (eps_t * eps_t)
            denominator = (ν - 2.0) + (eps_t * eps_t)
            score_factor = numerator / denominator 
            f[t + 1] = ω + α * score_factor * ft + β * ft
            
            # ensure positivity (numerical safeguard)
            if f[t + 1] <= 0:
                return 1e10, None
        
        # We also need to include the log‐likelihood contribution at t = T-1 (last point):
        y_last = self.y[-1]
        f_last = f[-1]
        eps_last = y_last / np.sqrt(f_last)
        logpdf_last = (
            const_part 
            - 0.5 * np.log(f_last)
            - ((ν + 1.0) / 2.0) 
              * np.log( 1.0 + (y_last * y_last) / ((ν - 2.0) * f_last) )
        )
        nll -= logpdf_last
        
        return nll, f
    def fit(self):
        """
        Estimate (ω, α, β) by minimizing the negative log‐likelihood.

        """
        # Initial guess: set ω ≈ 0.1 × Var(y), α = 0.05, β = 0.9
        sample_var = np.var(self.y)
        init_params = np.array([0.1, 0.1, 0.8, 6.0])  # [ω, α, β, ν]
        
        # Box‐bounds: ω ∈ (1e-8, ∞), α ∈ [0, 1), β ∈ [0, 1)
        bounds = [
            (1e-8, None),   # ω > 0
            (0.0, 0.9999),  # 0 ≤ α < 1
            (0.0, 0.9999) , 
              (2, 100)  # 0 ≤ β < 1
        ]
        
        def objective(p):
            nll, _ = self._compute_nll_and_variances(p)
            return nll

        result = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': False, 'maxiter': 1000}
        )
        
        self.params = result.x
        _, self.fitted_f = self._compute_nll_and_variances(self.params)
        return result

    def get_fitted_variances(self):
        """
        After fitting, returns the in‐sample sequence {f_t}.
        """
        return self.fitted_f

class Beta_t_EGARCH11:
    """
    Purpose
    -------
    Estimate a Beta-t-EGARCH(1,1) model by maximum likelihood.

    Model
    -----
    y_t = exp(lambda_{t|t-1} / 2) * epsilon_t,   epsilon_t ~ t_nu  (nu > 2)

    The conditional log-variance recursion is score-driven:

        lambda_{t+1|t} = omega
                         + alpha * u_t
                         + beta  * lambda_{t|t-1}

    where the score variable u_t is

        u_t = (nu + 1) * y_t^2 / [(nu - 2) * h_{t|t-1} + y_t^2]  -  1,

        h_{t|t-1} = exp(lambda_{t|t-1}).

    Parameters
    ----------
    vY : ndarray
        Return series (demeaned).

    Attributes
    ----------
    params : ndarray
        Estimated parameters (omega, alpha, beta, nu).

    fitted_f : ndarray
        In-sample conditional variance path h_{t|t-1} = exp(lambda_{t|t-1}).

    fitted_lnf : ndarray
        In-sample log-conditional-variance path lambda_{t|t-1}

    """

    def __init__(self, vY):

        self.vY = np.asarray(vY, dtype=float)
        self.iT = len(self.vY)

        self.params  = None
        self.fitted_f   = None
        self.fitted_lnf = None

    # =========================================================================
    # Parameter transforms
    # =========================================================================

    def _link(self, vParams):
        """
        Map structural parameters (omega, alpha, beta, nu) to the unconstrained
        space used by the optimiser.

        Transforms
        ----------
        omega : identity  (log-variance intercept; unrestricted in R)
        alpha : identity  (score loading; unrestricted in R, sign free)
        beta  : logit((beta + 1) / 2)
                Maps beta in (-1, 1) bijectively onto R.
                Inverse: beta = 2 * expit(z_beta) - 1.
                Standard logit / expit only covers (0, 1), so using them
                directly would prevent beta < 0, which is incorrect for EGARCH.
        nu    : logit((nu - 2) / nu_max)
                Maps nu in (2, 2 + nu_max) bijectively onto R.
                Inverse: nu = nu_max * expit(z_nu) + 2.
                nu_max = 100 keeps nu numerically bounded during optimisation.
        """
        dOmega, dAlpha, dBeta, dNu = vParams

        dNuMax = 100.0

        return np.array([
            dOmega,
            dAlpha,
            logit((dBeta + 1.0) / 2.0),           # beta in (-1, 1)
            logit((dNu - 2.0) / dNuMax)            # nu  in (2, 102)
        ])

    def _link_inverse(self, vParamsTr):
        """
        Inverse of _link: recover structural parameters from unconstrained vector.

        Inverse transforms
        ------------------
        omega : identity
        alpha : identity
        beta  : 2 * expit(z_beta) - 1   in (-1, 1)
        nu    : nu_max * expit(z_nu) + 2 in (2, 2 + nu_max)
        """
        dNuMax = 100.0

        dOmega = vParamsTr[0]
        dAlpha = vParamsTr[1]
        dBeta  = 2.0 * expit(vParamsTr[2]) - 1.0  # in (-1, 1)
        dNu    = dNuMax * expit(vParamsTr[3]) + 2.0

        return np.array([dOmega, dAlpha, dBeta, dNu])

    # =========================================================================
    # Filter and likelihood
    # =========================================================================

    def _compute_nll_and_filter(self, vParams):
        """
        Purpose
        -------
        Run the Beta-t-EGARCH(1,1) filter and return the negative
        log-likelihood together with the variance and log-variance paths.

        Parameters
        ----------
        vParams : array_like
            Structural parameters (omega, alpha, beta, nu).

        Returns
        -------
        dNLL : float
            Negative log-likelihood (sum, not mean, for comparability with
            Beta_t_GARCH11).

        vH : ndarray of shape (iT,)
            Conditional variance path h_{t|t-1} = exp(lambda_{t|t-1}).

        vLnH : ndarray of shape (iT,)
            Log-conditional-variance path lambda_{t|t-1}.
        """
        dOmega, dAlpha, dBeta, dNu = vParams

        # Initialise log-variance at unconditional mean
        if abs(dBeta) < 0.999:
            dLnH0 = dOmega / (1.0 - dBeta)
        else:
            dLnH0 = np.log(max(np.var(self.vY), 1e-8))

        vLnH    = np.empty(self.iT)
        vH      = np.empty(self.iT)
        vLnH[0] = dLnH0
        vH[0]   = np.exp(dLnH0)

        # Precompute Student-t log-density constant
        # log p(y | h) = C - 0.5*ln(h) - ((nu+1)/2)*ln(1 + y^2/((nu-2)*h))
        dC = (gammaln((dNu + 1.0) / 2.0)
              - gammaln(dNu / 2.0)
              - 0.5 * np.log(np.pi * (dNu - 2.0)))

        dNLL = 0.0

        for t in range(self.iT):

            dHt  = vH[t]
            dYt  = self.vY[t]

            # --- log-likelihood contribution at time t
            dNLL -= (dC
                     - 0.5 * vLnH[t]
                     - 0.5 * (dNu + 1.0) * np.log(1.0 + dYt**2 / ((dNu - 2.0) * dHt)))

            # --- score variable u_t (Harvey-Chakravarty eq. 18)
            #     u_t = (nu+1)*y_t^2 / [(nu-2)*h_t + y_t^2]  - 1
            dUt = (dNu + 1.0) * dYt**2 / ((dNu - 2.0) * dHt + dYt**2) - 1.0

            # --- update log-variance (only if t+1 exists)
            if t < self.iT - 1:
                vLnH[t + 1] = dOmega + dAlpha * dUt + dBeta * vLnH[t]
                vH[t + 1]   = np.exp(vLnH[t + 1])

                if not np.isfinite(vH[t + 1]) or vH[t + 1] <= 0:
                    return 1e10, None, None

        return dNLL, vH, vLnH

    def _nll_tr(self, vParamsTr):
        """
        Negative log-likelihood in the unconstrained parameterisation,
        for use by the optimiser.
        """
        vParams = self._link_inverse(vParamsTr)
        dNLL, _, _ = self._compute_nll_and_filter(vParams)
        return dNLL

    # =========================================================================
    # Estimation
    # =========================================================================

    def fit(self):
        """
        Purpose
        -------
        Estimate (omega, alpha, beta, nu) by minimising the negative
        log-likelihood via L-BFGS-B in the unconstrained parameterisation.

        Returns
        -------
        self
        """
        dVarY   = np.var(self.vY)
        dBeta0  = 0.97
        dOmega0 = np.log(max(dVarY, 1e-8)) * (1.0 - dBeta0)

        vStart = np.array([
            dOmega0,   # omega
            0.05,      # alpha  (positive: larger shocks raise log-variance)
            dBeta0,    # beta
            8.0        # nu
        ])

        vStartTr = self._link(vStart)

        oRes = minimize(
            self._nll_tr,
            vStartTr,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'disp': False}
        )

        vParams = self._link_inverse(oRes.x)
        _, vH, vLnH = self._compute_nll_and_filter(vParams)

        self.params    = vParams
        self.fitted_f  = vH
        self.fitted_lnf = vLnH

        return self

    # =========================================================================
    # Accessors
    # =========================================================================

    def get_fitted_variances(self):
        """
        Purpose
        -------
        Return the in-sample conditional variance path after fitting.

        Returns
        -------
        ndarray
            h_{t|t-1} for t = 1, ..., T.
        """
        return self.fitted_f

    def forecast_variance(self, horizon=1):
        """
        Purpose
        -------
        Multi-step-ahead conditional variance forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        ndarray of shape (horizon,)
            Variance forecasts h_{T+h|T} for h = 1, ..., horizon.

        Notes
        -----
        At h = 1 the last observed u_T is known, so the forecast uses the
        exact recursion. For h > 1, E[u_{T+j}] = 0 (u_t is a martingale
        difference), so the log-variance mean-reverts geometrically:

            E[lambda_{T+h|T}] = omega/(1-beta)
                                 + beta^(h-1) * (lambda_{T+1|T} - omega/(1-beta))

        Forecasts are returned on the variance scale via exponentiation of
        the expected log-variance; this gives the median rather than the mean
        forecast, consistent with Beta_t_GARCH11 and EGARCH11.
        """
        dOmega, dAlpha, dBeta, dNu = self.params

        dLnH_T   = self.fitted_lnf[-1]
        dH_T     = self.fitted_f[-1]
        dY_T     = self.vY[-1]

        # Exact u_T
        dU_T = (dNu + 1.0) * dY_T**2 / ((dNu - 2.0) * dH_T + dY_T**2) - 1.0

        # h = 1: exact one-step-ahead log-variance
        dLnH_T1 = dOmega + dAlpha * dU_T + dBeta * dLnH_T

        vForecasts = np.empty(horizon)
        vForecasts[0] = np.exp(dLnH_T1)

        # h > 1: E[u_{T+j}] = 0, so log-variance follows AR(1) mean-reversion
        dUncond = dOmega / (1.0 - dBeta) if abs(dBeta) < 0.999 else dLnH_T1

        for h in range(1, horizon):
            dLnH_next = dUncond + dBeta**h * (dLnH_T1 - dUncond)
            vForecasts[h] = np.exp(dLnH_next)

        return vForecasts
    


class BM_GARCH11:
    """
    Purpose
    -------
    Estimate a GARCH(1,1) model by the bounded M-estimate (BM-estimate) of
    Muler and Yohai (2008).

    Model
    -----
    x_t = sigma_t * z_t,   z_t i.i.d.,  E[z_t] = 0,  Var(z_t) = 1

    Ordinary recursion (used for the M-estimate lambda_1), their eq. (7):

        sigma_t^2 = omega + alpha * x_{t-1}^2 + beta * sigma_{t-1}^2.

    Filtered recursion (used for the M-estimate lambda_2), their eq. (14),
    which bounds the propagation of an outlier through the conditional-variance
    predictor:

        sigma*_t^2 = omega
                     + alpha * sigma*_{t-1}^2 * r_k( x_{t-1}^2 / sigma*_{t-1}^2 )
                     + beta  * sigma*_{t-1}^2,

        r_k(u) = u if u <= k, else k.

    Note that the ratio x_{t-1}^2 / sigma*_{t-1}^2 here is the argument of the
    clip r_k in eq. (14); it is a genuine part of the model and is unrelated to
    the change-of-variable used below to write the loss.

    Estimation (paper's log scale, their eqs. 9-12)
    -----------------------------------------------
    Put y_t = log x_t^2 and h_t(c) = log sigma_t^2, and let u_t = y_t - h_t(c).
    The Gaussian QML objective is the M-objective mean_t rho_0(u_t) with the
    unbounded loss rho_0 = -log g_0 (their eq. 6, 11):

        g_0(w)   = (1 / sqrt(2 pi)) * exp( -0.5 * (exp(w) - w) ),
        rho_0(u) = 0.5 * exp(u) - 0.5 * u + 0.5 * log(2 pi).

    (The identity exp(u_t) = x_t^2 / sigma_t^2 recovers the variance-scale form
    of eq. (8); we keep the log-scale form above, as stated in the paper.)

    Robustness on the loss is obtained with the smoothed cap m_1 of Section 4,
    so rho(u) = m_1(rho_0(u)). With the normal-based rho_0 the Fisher
    correction u_0 is zero (Muler and Yohai 2008, Thm. 1(ii)), so no correction
    is applied.

    Two M-estimates are computed: lambda_1 minimizing the capped objective with
    the ordinary recursion, and lambda_2 minimizing the capped objective with the
    filtered recursion. The BM-estimate selects between them (their eq. 18):

        lambda_B = lambda_1 if M(lambda_1) <= M*(lambda_2), else lambda_2,

    where M is the ordinary-recursion objective at lambda_1 and M* is the
    filtered-recursion objective at lambda_2. Under a clean GARCH the two
    coincide asymptotically (their Thm. 4), recovering the efficient M-estimate;
    under contamination the filtered fit is selected.

    Smoothed cap m_1 (their Section 4), transcribed verbatim from the paper:

        m_1(x) = x        if x <= a,
                 P(x)     if a < x <= b,
                 4.15     if x > b,

    with a = 4.0, b = 4.3, and

        P(x) = (2 / (b-a)^3) * [ 0.25*(x^4 - a^4)
                                 - (1/3)*(2a+b)*(x^3 - a^3)
                                 + 0.5*(a^2 + 2ab)*(x^2 - a^2) ]
               - (2 a^2 b / (b-a)^3) * (x - a)
               - (1 / (3 (b-a)^2)) * (x - a)^3
               + x.

    The last cubic term is the one ambiguous piece of the printed formula; it is
    read here as 1 / (3 (b-a)^2). The plateau value for x > b is the paper's
    stated 4.15 (parameter dCapPlateau).

    Parameters
    ----------
    vY : ndarray
        Return series.
    dCapLow : float
        Lower knot a of m_1 (where smoothing begins; BM1 default 4.0).
    dCapHigh : float
        Upper knot b of m_1 (where P ends and the plateau begins; BM1 default 4.3).
    dCapPlateau : float
        Value of m_1(x) for x > b, as printed in the paper (BM1 default 4.15).
    dK : float
        Clipping constant k of the filtered recursion (BM1 default 5.02).

    Attributes
    ----------
    params : tuple
        Selected parameters (omega, alpha, beta).
    fitted_f : ndarray
        Filtered conditional variance path under the selected recursion.
    bRobustSelected : bool
        True if the filtered (lambda_2) branch was selected.
    lambda1, lambda2 : tuple
        The two underlying M-estimates.
    M1, M2 : float
        Their achieved objective values M(lambda_1) and M*(lambda_2).
    """

    def __init__(self, vY, dCapLow=4.0, dCapHigh=4.3, dCapPlateau=4.15, dK=5.02):

        self.vY = np.asarray(vY, dtype=float)

        self.dCapLow = float(dCapLow)
        self.dCapHigh = float(dCapHigh)
        self.dCapPlateau = float(dCapPlateau)
        self.dK = float(dK)

        # y_t = log x_t^2, floored for numerical safety at exact/near zeros
        self.vLogY2 = np.log(np.maximum(self.vY**2, 1e-12))

        self.params = None
        self.fitted_f = None
        self.bRobustSelected = None

    def _m1(self, vV):
        """
        Smoothed cap m_1 of Muler-Yohai (2008), Section 4, using their printed
        P(x) verbatim.

        Parameters
        ----------
        vV : ndarray
            Loss values rho_0 to be capped.

        Returns
        -------
        ndarray
            m_1(vV): identity below dCapLow, the paper's polynomial P on
            [dCapLow, dCapHigh], and the constant dCapPlateau above.
        """

        dA, dB = self.dCapLow, self.dCapHigh

        vV = np.asarray(vV, dtype=float)
        vOut = np.empty_like(vV)

        mLow = vV <= dA
        mHigh = vV > dB
        mMid = ~(mLow | mHigh)

        vOut[mLow] = vV[mLow]
        vOut[mHigh] = self.dCapPlateau

        x = vV[mMid]
        dD3 = (dB - dA)**3
        dD2 = (dB - dA)**2

        vOut[mMid] = (
            (2.0 / dD3) * (
                0.25 * (x**4 - dA**4)
                - (1.0 / 3.0) * (2.0 * dA + dB) * (x**3 - dA**3)
                + 0.5 * (dA**2 + 2.0 * dA * dB) * (x**2 - dA**2)
            )
            - (2.0 * dA**2 * dB / dD3) * (x - dA)
            - (1.0 / (3.0 * dD2)) * (x - dA)**3   # ambiguous term: 1 / (3 (b-a)^2)
            + x
        )

        return vOut

    def _filter(self, vParams):
        """
        Ordinary GARCH(1,1) variance recursion (eq. 7).

        Parameters
        ----------
        vParams : array_like
            (omega, alpha, beta).

        Returns
        -------
        ndarray
            Conditional variance path.
        """

        dOmega, dAlpha, dBeta = vParams

        iT = len(self.vY)

        vH = np.zeros(iT)

        vH[0] = dOmega / max(1.0 - dAlpha - dBeta, 1e-6)

        for t in range(1, iT):

            vH[t] = (
                dOmega
                + dAlpha * self.vY[t - 1]**2
                + dBeta * vH[t - 1]
            )

        return vH

    def _filter_robust(self, vParams):
        """
        Filtered GARCH(1,1) variance recursion with clipped standardized
        innovation, r_k(x_{t-1}^2 / sigma*_{t-1}^2) (eq. 14).

        Parameters
        ----------
        vParams : array_like
            (omega, alpha, beta).

        Returns
        -------
        ndarray
            Filtered conditional variance path sigma*_t^2.
        """

        dOmega, dAlpha, dBeta = vParams

        iT = len(self.vY)

        vH = np.zeros(iT)

        vH[0] = dOmega / max(1.0 - dAlpha - dBeta, 1e-6)

        for t in range(1, iT):

            dPrev = vH[t - 1]

            dRatio = self.vY[t - 1]**2 / dPrev

            dRk = dRatio if dRatio <= self.dK else self.dK

            vH[t] = (
                dOmega
                + dAlpha * dPrev * dRk
                + dBeta * dPrev
            )

        return vH

    def _objective(self, vParams, bRobust):
        """
        Mean capped M-objective mean_t m_1(rho_0(u_t)), with the loss written on
        the paper's log scale u_t = y_t - log sigma_t^2 (eqs. 9-12).

        Parameters
        ----------
        vParams : array_like
            (omega, alpha, beta).
        bRobust : bool
            If True, use the filtered recursion; otherwise the ordinary one.

        Returns
        -------
        float
            Mean capped loss, or 1e10 on a non-stationary or degenerate path.
        """

        dOmega, dAlpha, dBeta = vParams

        # stationarity (compact set C of Muler-Yohai, alpha + beta < 1)
        if dAlpha + dBeta >= 1.0:
            return 1e10

        vH = self._filter_robust(vParams) if bRobust else self._filter(vParams)

        if np.any(vH <= 0) or np.any(~np.isfinite(vH)):
            return 1e10

        # u_t = y_t - log sigma_t^2,  y_t = log x_t^2
        vU = self.vLogY2 - np.log(vH)

        # rho_0(u) = 0.5 exp(u) - 0.5 u + 0.5 log(2 pi)  = -log g_0(u)
        vRho0 = 0.5 * np.exp(vU) - 0.5 * vU + 0.5 * np.log(2.0 * np.pi)

        vRho = self._m1(vRho0)

        return np.mean(vRho)

    def fit(self):
        """
        Compute lambda_1 (ordinary recursion) and lambda_2 (filtered recursion),
        then select the BM-estimate by Muler-Yohai eq. (18).

        Returns
        -------
        self
        """

        vStart = np.array([0.2, 0.1, 0.80])

        lBounds = [
            (1e-8, None),
            (0.0, 0.9999),
            (0.0, 0.9999)
        ]

        # --- M-estimate with the ordinary recursion
        oRes1 = minimize(
            lambda p: self._objective(p, False),
            vStart,
            method="L-BFGS-B",
            bounds=lBounds
        )
        vLam1 = oRes1.x
        dM1 = self._objective(vLam1, False)

        # --- M-estimate with the filtered recursion
        oRes2 = minimize(
            lambda p: self._objective(p, True),
            vStart,
            method="L-BFGS-B",
            bounds=lBounds
        )
        vLam2 = oRes2.x
        dM2 = self._objective(vLam2, True)

        # --- BM selection (eq. 18): ordinary if it achieves the lower objective
        if dM1 <= dM2:
            self.params = tuple(vLam1)
            self.bRobustSelected = False
            self.fitted_f = self._filter(vLam1)
        else:
            self.params = tuple(vLam2)
            self.bRobustSelected = True
            self.fitted_f = self._filter_robust(vLam2)

        self.lambda1 = tuple(vLam1)
        self.lambda2 = tuple(vLam2)
        self.M1 = dM1
        self.M2 = dM2

        return self

    def get_fitted_variances(self):

        return self.fitted_f

    def forecast_variance(self, horizon=1):
        """
        Multi-step variance forecasts under the selected recursion.

        Notes
        -----
        h = 1 is exact (x_T is observed): the ordinary branch uses
        omega + alpha x_T^2 + beta sigma_T^2, the filtered branch uses
        omega + alpha sigma*_T^2 r_k(x_T^2 / sigma*_T^2) + beta sigma*_T^2.
        For h > 1 the log-variance mean-reverts geometrically with persistence
        alpha + beta toward omega / (1 - alpha - beta), matching the other
        benchmark classes.
        """

        dOmega, dAlpha, dBeta = self.params

        dPersistence = dAlpha + dBeta

        dHT = self.fitted_f[-1]
        dYT = self.vY[-1]

        vForecasts = np.zeros(horizon)

        if self.bRobustSelected:
            dRatio = dYT**2 / dHT
            dRk = dRatio if dRatio <= self.dK else self.dK
            dH1 = dOmega + dAlpha * dHT * dRk + dBeta * dHT
        else:
            dH1 = dOmega + dAlpha * dYT**2 + dBeta * dHT

        vForecasts[0] = dH1

        dUncond = dOmega / (1.0 - dPersistence) if dPersistence < 1.0 else dH1

        for h in range(1, horizon):
            vForecasts[h] = dUncond + dPersistence**h * (dH1 - dUncond)

        return vForecasts
