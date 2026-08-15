import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


class BLADEFilter:
    """
    BLADE filter for a conditional-variance or conditional-location model
    (BLADE paper). Estimation minimises the average Barron scoring rule over an
    unconstrained reparametrisation of (omega, alpha, beta), using numerical
    (finite-difference) gradients.

    For the volatility model with dGamma <= 2 the recursion is written in the
    regrouped, guaranteed-positive form
        vartheta_t = omega + alpha * w(x) * y^2 + vartheta_{t-1} * (beta - alpha * w(x)),
    where w(x) = psi(x)/x is the bounded Barron weight with sup_x w = 1/dXi**2.
    The parameter transform then enforces the sufficient positivity condition
    beta >= alpha / dXi**2 via a log-gap, so vartheta_t >= omega > 0 holds by
    induction for every realisation.

    Parameters
    ----------
    sModelType : string,  'volatility' or 'location'
    dGamma     : double,  Barron shape parameter (gamma; 2 = Gaussian)
    dXi        : double,  Barron scale parameter (xi)
    """

    def __init__(self, sModelType: str = 'volatility', dGamma: float = 1.5, dXi: float = 1.0):
        """
        Purpose
        ----------
        Initialise the BLADE filter and store its fixed (non-estimated) settings.

        Parameters
        ----------
        sModelType :   string, model variant, 'volatility' or 'location'
        dGamma :       double, Barron shape parameter (2 = Gaussian score)
        dXi :          double, Barron scale parameter

        Returns
        -------
        None
        """
        if sModelType not in ['volatility', 'location']:
            raise ValueError("sModelType must be either 'volatility' or 'location'")

        self.sModelType = sModelType
        self.dGamma = dGamma
        self.dXi = dXi

        self.param_names = ['omega', 'alpha', 'beta']
        self.params = None
        self.vFitted = None
        self.vResiduals = None

        # Negative-variance diagnostics (volatility only; reset each filter call)
        self.iNegCount = 0
        self.dNegMin = 0.0
        self.iFilterT = 0

    def _dSupW(self) -> float:
        """
        Purpose
        ----------
        Supremum of the Barron weight w(x) = psi(x)/x. For dGamma <= 2 the
        weight is maximised at x = 0 with value 1/dXi**2.

        Returns
        -------
        dSupW :   double, sup_x w(x)
        """
        return 1.0 / self.dXi**2

    def _S(self, vX: np.ndarray, dGamma: float, dXi: float) -> np.ndarray:
        """
        Purpose
        ----------
        Barron scoring rule S(x), evaluated elementwise.

        Parameters
        ----------
        vX :       array or double, argument of the scoring rule (the residual x)
        dGamma :   double, Barron shape parameter (2 = Gaussian)
        dXi :      double, Barron scale parameter

        Returns
        -------
        vS :       array or double, scoring rule value S(x)
        """
        if dGamma == 2:
            return 0.5 * (vX / dXi)**2
        elif dGamma == 0:
            return np.log(1 + 0.5 * (vX / dXi)**2)
        elif dGamma == float('-inf'):
            return 1.0 - np.exp(-0.5 * (vX / dXi)**2)
        else:
            dA = np.abs(dGamma - 2)
            return (dA / dGamma) * ((1 + (vX**2) / (dXi**2 * dA))**(dGamma / 2) - 1)

    def _psi(self, vX: np.ndarray, dGamma: float, dXi: float) -> np.ndarray:
        """
        Purpose
        ----------
        Barron score psi(x) = S'(x), evaluated elementwise. This is the
        derivative of the scoring rule and is the term that drives the filter
        update.

        Parameters
        ----------
        vX :       array or double, argument of the score (the residual x)
        dGamma :   double, Barron shape parameter (2 = Gaussian)
        dXi :      double, Barron scale parameter

        Returns
        -------
        vPsi :     array or double, score value psi(x) = S'(x)
        """
        if dGamma == 2:
            return vX / (dXi**2)
        elif dGamma == 0:
            return (2 * vX) / (vX**2 + 2 * dXi**2)
        elif dGamma == float('-inf'):
            return (vX / dXi**2) * np.exp(-0.5 * (vX / dXi)**2)
        else:
            dScale = (dXi**2) * np.abs(dGamma - 2)
            vDenom = 1 + (vX**2) / dScale
            dExponent = 1 - (dGamma / 2)
            return (vX / dXi**2) * (vDenom ** (-dExponent))

    def _psi_weight(self, vX: np.ndarray, dGamma: float, dXi: float) -> np.ndarray:
        """
        Purpose
        ----------
        Barron weight w(x) = psi(x)/x, the bounded multiplier on the raw
        residual. For dGamma < 2, 0 < w(x) <= 1/dXi**2, with the sup at x = 0.
        Defined by continuity at x = 0 (w(0) = 1/dXi**2), so it is safe for the
        regrouped recursion.

        Parameters
        ----------
        vX :       array or double, raw residual x
        dGamma :   double, Barron shape parameter
        dXi :      double, Barron scale parameter

        Returns
        -------
        vW :       array or double, weight w(x)
        """
        if dGamma == 2:
            return np.full_like(np.asarray(vX, dtype=float), 1.0 / dXi**2)
        elif dGamma == 0:
            # psi/x = 2 / (x^2 + 2 xi^2)
            return 2.0 / (vX**2 + 2 * dXi**2)
        elif dGamma == float('-inf'):
            return (1.0 / dXi**2) * np.exp(-0.5 * (vX / dXi)**2)
        else:
            dScale = (dXi**2) * np.abs(dGamma - 2)
            vDenom = 1 + (vX**2) / dScale
            return (1.0 / dXi**2) * (vDenom ** (dGamma / 2 - 1))

    def param_transform(self, vParams: np.ndarray) -> np.ndarray:
        """
        Purpose
        ----------
        Map natural (omega, alpha, beta) to unconstrained R^3.
        Volatility: omega -> log(omega) (omega > 0); alpha -> log(alpha)
        (alpha > 0); beta -> log of the positivity gap, beta = alpha*supW +
        exp(delta), which enforces beta >= alpha*supW.
        Location: omega identity (omega in R); alpha -> log(alpha);
        beta -> logit(beta) (beta in (0,1)); no positivity constraint needed.

        Parameters
        ----------
        vParams :    array, natural parameters [omega, alpha, beta]

        Returns
        -------
        vParamsTr :  array, transformed (free) parameters
        """
        dOmega, dAlpha, dBeta = vParams[0], vParams[1], vParams[2]
        if self.sModelType == 'location':
            return np.array([dOmega, np.log(dAlpha), logit(dBeta)])
        dSupW = self._dSupW()
        dGap = dBeta - dAlpha * dSupW
        if dGap <= 0:
            raise ValueError(f"Initial beta - alpha*supW = {dGap:.3e} <= 0; "
                             f"positivity constraint violated at init.")
        return np.array([np.log(dOmega), np.log(dAlpha), np.log(dGap)])

    def inverse_param_transform(self, vParamsTr: np.ndarray) -> np.ndarray:
        """
        Purpose
        ----------
        Inverse of param_transform; map the free parameters back to the natural
        (omega, alpha, beta).

        Parameters
        ----------
        vParamsTr :  array, transformed (free) parameters

        Returns
        -------
        vParams :    array, natural parameters [omega, alpha, beta]
        """
        if self.sModelType == 'location':
            return np.array([vParamsTr[0], np.exp(vParamsTr[1]), expit(vParamsTr[2])])
        dSupW = self._dSupW()
        dOmega = np.exp(vParamsTr[0])
        dAlpha = np.exp(vParamsTr[1])
        dBeta = dAlpha * dSupW + np.exp(vParamsTr[2])
        return np.array([dOmega, dAlpha, dBeta])

    def filter(self, vY: np.ndarray, vParams: np.ndarray, bRegrouped: bool = True) -> np.ndarray:
        """
        Purpose
        ----------
        Run the BLADE filter recursion. For volatility with bRegrouped=True the
        guaranteed-positive form is used:
            vartheta_t = omega + alpha*w(x)*y^2 + vartheta_{t-1}*(beta - alpha*w(x)),
        which is algebraically identical to the raw form
            vartheta_t = omega + alpha*psi(x) + beta*vartheta_{t-1},
        with x = y_{t-1}^2 - vartheta_{t-1}. Location always uses the raw form.

        Parameters
        ----------
        vY :          array, time series data
        vParams :     array, natural parameters [omega, alpha, beta]
        bRegrouped :  boolean, use regrouped positive form (volatility only)

        Returns
        -------
        vVartheta :   array, filtered conditional variance (volatility) or
                      conditional mean (location) path
        """
        dOmega, dAlpha, dBeta = vParams[0], vParams[1], vParams[2]
        iT = len(vY)

        vVartheta = np.zeros(iT)
        vVartheta[0] = np.var(vY) if self.sModelType == 'volatility' else np.mean(vY)

        for it in range(1, iT):
            if self.sModelType == 'volatility':
                dX = vY[it-1]**2 - vVartheta[it-1]
                if bRegrouped:
                    dW = self._psi_weight(dX, self.dGamma, self.dXi)
                    vVartheta[it] = (dOmega
                                     + dAlpha * dW * vY[it-1]**2
                                     + vVartheta[it-1] * (dBeta - dAlpha * dW))
                else:
                    vVartheta[it] = (dOmega
                                     + dAlpha * self._psi(dX, self.dGamma, self.dXi)
                                     + dBeta * vVartheta[it-1])
            else:
                dX = vY[it-1] - vVartheta[it-1]
                vVartheta[it] = (dOmega
                                 + dAlpha * self._psi(dX, self.dGamma, self.dXi)
                                 + dBeta * vVartheta[it-1])

        self.iFilterT = iT
        if self.sModelType == 'volatility':
            vNeg = vVartheta[vVartheta < 0]
            self.iNegCount = vNeg.size
            self.dNegMin = vNeg.min() if vNeg.size else 0.0
        return vVartheta

    def objective_function(self, vParamsTr: np.ndarray, vY: np.ndarray) -> float:
        """
        Purpose
        ----------
        Average Barron scoring rule L_T = (1/T) sum_t S(x_t), as a function of
        the free (transformed) parameters.

        Parameters
        ----------
        vParamsTr :  array, transformed (free) parameters
        vY :         array, time series data

        Returns
        -------
        dLoss :      double, mean scoring rule (returns 1e10 if non-finite)
        """
        vParams = self.inverse_param_transform(vParamsTr)
        vVartheta = self.filter(vY, vParams)
        vResiduals = (vY**2 - vVartheta) if self.sModelType == 'volatility' else (vY - vVartheta)
        dLoss = np.mean(self._S(vResiduals, self.dGamma, self.dXi))
        return dLoss if np.isfinite(dLoss) else 1e10

    def param_estimate(self, vY: np.ndarray, iMaxiter: int = 2000) -> dict:
        """
        Purpose
        ----------
        Estimate (omega, alpha, beta), then store the fitted path and residuals.

        Parameters
        ----------
        vY :         array, time series data
        iMaxiter :   integer, maximum number of optimiser iterations

        Returns
        -------
        params :     dict, estimated natural parameters keyed by name
                     {'omega', 'alpha', 'beta'}
        """
        # alpha's init must satisfy alpha*supW < beta (supW = 1/dXi**2 for
        # volatility) or param_transform raises at the very first call. 0.1
        # is feasible against beta=0.8 at this folder's dXi=0.7 (gap=0.596),
        # but scaling by dXi**2 keeps alpha*supW = 0.1 regardless of dXi --
        # identical init to the dXi=1 case and safe against this constraint
        # for any xi, not just the one this folder happens to fix.
        dAlphaInit = 0.1*self.dXi**2 if self.sModelType == 'volatility' else 0.1
        vInit = np.array([0.2, dAlphaInit, 0.8])
        vInitTr = self.param_transform(vInit)

        oResult = minimize(self.objective_function, vInitTr, args=(vY,),
                           method="L-BFGS-B", options={'maxiter': iMaxiter})
        if not oResult.success:
            print(f"Warning: optimizer did not converge cleanly: {oResult.message}")

        vFinalParams = self.inverse_param_transform(oResult.x)
        self.params = {sName: dVal for sName, dVal in zip(self.param_names, vFinalParams)}
        self.vFitted = self.filter(vY, vFinalParams)
        self.vResiduals = (vY**2 - self.vFitted) if self.sModelType == 'volatility' else (vY - self.vFitted)

        if self.sModelType == 'volatility' and self.iNegCount > 0:
            print(f"Warning: conditional variance went negative on "
                  f"{self.iNegCount}/{self.iFilterT} steps, most negative = {self.dNegMin:.3e}")
        return self.params