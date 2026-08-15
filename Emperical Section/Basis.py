#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RiskBasis: building blocks for the parametric BLADE Value-at-Risk analysis.

This module holds every reusable statistical/forecasting piece: per-window
forecasts, the rolling driver, the Student-t innovation fit and its VaR/ES
multipliers, backtests, the full analysis routine, the log-score
Diebold-Mariano test, and the second-stage rolling window. Plotting lives in
the sibling module Plots.py. The entry point that
wires it all together and writes the CSVs/figures lives in RiskMain.py.

Models are imported:
    BLADEFilter        from BLADE_Filter          
    Beta_t_GARCH11, Beta_t_EGARCH11,GARCH11, BM_GARCH11, Student_t_GARCH11 from Benchmark_models


Notation:
  p       exceedance (tail) probability of the VaR/ES level, e.g. p = 0.01.
  dGamma  Barron robustness parameter of the BLADE filter (kept as in the
          BLADEFilter API); the BLADE model columns are named BLADE_g{dGamma}.
  h       conditional variance forecast (standardised unless rescaled by the
          per-window scale, in which case it is on original units).

Author: Mathijs Dijkstra
Date:   July 2026 -  Updated: August 2026
"""

# --- packages
import warnings
warnings.filterwarnings('ignore')

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, t as student_t
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# --- models (keep the local model directory importable)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BLADE_Filter import BLADEFilter
from Benchmark_models import Beta_t_GARCH11, GARCH11, Beta_t_EGARCH11, Student_t_GARCH11, BM_GARCH11


# ===========================================================================
# 1.  Per-window one-step variance forecasts
# ===========================================================================
def garch_forecast_window(vTrain):
    """
    Purpose
    -------
    Produce a one-step-ahead GARCH(1,1) variance forecast.

    Parameters
    ----------
    vTrain : ndarray
        Standardized estimation window.

    Returns
    -------
    float
        Forecasted conditional variance h_{T+1|T}.
    """
    oM = GARCH11(vTrain)
    oM.fit()
    dOmega, dAlpha, dBeta = oM.params
    vH = oM.fitted_f
    return max(dOmega + dAlpha*vTrain[-1]**2 + dBeta*vH[-1], 1e-8)


def bm_garch_forecast_window(vTrain):
    """
    Purpose
    -------
    Produce a one-step-ahead BM-GARCH(1,1) variance forecast.

    Parameters
    ----------
    vTrain : ndarray
        Standardized estimation window.

    Returns
    -------
    float
        Forecasted conditional variance h_{T+1|T}.
    """
    oM = BM_GARCH11(vTrain)
    oM.fit()
    dOmega, dAlpha, dBeta = oM.params
    vH = oM.fitted_f
    return max(dOmega + dAlpha*vTrain[-1]**2 + dBeta*vH[-1], 1e-8)


def student_t_garch_forecast_window(vTrain):
    """
    Purpose
    -------
    Produce a one-step-ahead Student-t GARCH(1,1) variance forecast.

    Parameters
    ----------
    vTrain : ndarray
        Standardized estimation window.

    Returns
    -------
    float
        Forecasted conditional variance h_{T+1|T}.

    """
    oM = Student_t_GARCH11(vTrain)
    oM.fit()
    dOmega, dAlpha, dBeta, dNu = oM.params
    vH = oM.fitted_f
    return max(dOmega + dAlpha*vTrain[-1]**2 + dBeta*vH[-1], 1e-8)


def betat_forecast_window(vTrain):
    """
    Purpose
    -------
    Produce a one-step-ahead variance forecast from the beta-t-GARCH model.

    Parameters
    ----------
    vTrain : ndarray
        Standardized estimation window.

    Returns
    -------
    float
        Forecasted conditional variance h_{T+1|T}.
    """
    oM = Beta_t_GARCH11(vTrain)
    oM.fit()
    dO, dA, dB, dNu = oM.params
    vF = oM.fitted_f
    dEps = vTrain[-1]/np.sqrt(max(vF[-1], 1e-12))
    dScoreFac = (dNu + 1.0)*dEps**2/(dNu - 2.0 + dEps**2)
    return max(dO + dA*dScoreFac*vF[-1] + dB*vF[-1], 1e-8)


def betat_eg_forecast_window(vTrain):
    """
    Purpose
    -------
    Produce a one-step-ahead variance forecast from the Beta-t-EGARCH(1,1) model
    of Harvey and Chakravarty (2008).

    Parameters
    ----------
    vTrain : ndarray
        Standardized estimation window.

    Returns
    -------
    float
        Forecasted conditional variance h_{T+1|T} = exp(lambda_{T+1|T}).
    """
    oM = Beta_t_EGARCH11(vTrain)
    oM.fit()
    return max(oM.forecast_variance(horizon=1)[0], 1e-8)


def blade_forecast_window(vTrain, dGamma):
    """
    Purpose
    -------
    Estimate the BLADE filter and compute a one-step-ahead variance forecast.

    Parameters
    ----------
    vTrain : ndarray
        Standardized training sample.
    dGamma : float
        Barron robustness parameter.

    Returns
    -------
    float
        Forecasted conditional variance h_{T+1|T}.

    """
    m = BLADEFilter('volatility', dGamma=dGamma, dXi=0.6)
    m.param_estimate(vTrain)
    dX = vTrain[-1]**2 - m.vFitted[-1]
    return (m.params['omega']
            + m.params['alpha']*m._psi(dX, dGamma, 0.6)
            + m.params['beta']*m.vFitted[-1])


# ---------------------------------------------------------------------------
# Central benchmark registry. Every downstream loop (rolling step, diagnostics,
# tails, backtest and DM) keys off this single list, so adding or removing a
# benchmark only needs to happen here.
# ---------------------------------------------------------------------------
BENCH_MODELS = [
    ('GARCH',    garch_forecast_window),
    ('BETAT',    betat_forecast_window),
    ('BETAT_EG', betat_eg_forecast_window),
    ('BM',       bm_garch_forecast_window),
    ('STUDENTT', student_t_garch_forecast_window),
]
BENCH_NAMES = [sName for sName, _ in BENCH_MODELS]


# ===========================================================================
# 2.  One rolling step (parallel)
# ===========================================================================
def _one_step(iStep, vY, iTrain, vGammas):
    """
    Purpose
    -------
    Execute one rolling-window estimation and forecasting step.

    Parameters
    ----------
    iStep : int
        Starting index of the estimation window.
    vY : ndarray
        Full return series.
    iTrain : int
        Length of the estimation window.
    vGammas : list[float]
        BLADE Barron robustness parameters.

    Returns
    -------
    dict
        One-step-ahead conditional variance forecasts for every model, together
        with the per-window scale and the (standardised) realised next return.
    """
    warnings.filterwarnings('ignore')
    vTrainRaw = vY[iStep:iStep + iTrain]
    dScale    = np.std(vTrainRaw)
    vTrain    = vTrainRaw/dScale
    dRes = {'step': iStep, 'scale': dScale, 'y_next': vY[iStep + iTrain]/dScale}

    for dG in vGammas:
        try:
            dForecast = blade_forecast_window(vTrain, dG)
            if not np.isfinite(dForecast) or dForecast <= 0:
                raise ValueError(f"BLADE forecast non-positive for gamma={dG}")
            dRes[f'BLADE_g{dG}_hfore'] = max(dForecast, 1e-8)
        except Exception:
            dRes[f'BLADE_g{dG}_hfore'] = np.nan

    for sModel, fModel in BENCH_MODELS:
        try:
            dF = fModel(vTrain)
            dRes[f'{sModel}_hfore'] = dF if (np.isfinite(dF) and dF > 0) else np.nan
        except Exception:
            dRes[f'{sModel}_hfore'] = np.nan

    return dRes


def rolling_forecasts(vY, iTrain, vGammas, iNJobs=-1):
    """
    Purpose
    -------
    Generate rolling one-step-ahead variance forecasts for all competing models
    over the entire out-of-sample period.

    Parameters
    ----------
    vY : ndarray
        Full return series.
    iTrain : int
        Length of the estimation window.
    vGammas : list[float]
        BLADE Barron robustness parameters.
    iNJobs : int
        Number of parallel jobs.

    Returns
    -------
    pandas.DataFrame
        One row per rolling step, indexed by step, holding every model's
        forecast column, the scale, and the realised next return.
    """
    iT_oos = len(vY) - iTrain
    print(f"Rolling forecasts: T={len(vY)}, T_train={iTrain}, T_oos={iT_oos} ")
    lR = Parallel(n_jobs=iNJobs, verbose=5)(
        delayed(_one_step)(i, vY, iTrain, vGammas) for i in range(iT_oos))
    return pd.DataFrame(lR).set_index('step').sort_index()


# ===========================================================================
# 3.  Backtests and scoring (model-agnostic)
# ===========================================================================
def kupiec_uc(vHit, dP):
    """
    Purpose
    -------
    Kupiec unconditional coverage test.

    Parameters
    ----------
    vHit : ndarray
        Binary violation sequence (1 when the loss exceeds VaR).
    dP : float
        Nominal exceedance probability under the null.

    Returns
    -------
    float
        p-value of the likelihood-ratio statistic against chi-square with one
        degree of freedom, or NaN when the sample is empty.
    """
    iN = vHit.size; iX = int(vHit.sum())
    if iN == 0:
        return np.nan
    dPi = iX/iN
    if iX == 0 or iX == iN:
        dLR = -2*(iX*np.log(dP) + (iN - iX)*np.log(1 - dP))
    else:
        dLR = -2*((iX*np.log(dP) + (iN - iX)*np.log(1 - dP))
                  - (iX*np.log(dPi) + (iN - iX)*np.log(1 - dPi)))
    return 1 - chi2.cdf(dLR, 1)


def christoffersen_cc(vHit, dP):
    """
    Purpose
    -------
    Christoffersen conditional coverage test (joint unconditional coverage and
    first-order independence of violations).

    Parameters
    ----------
    vHit : ndarray
        Binary violation sequence (1 when the loss exceeds VaR).
    dP : float
        Nominal exceedance probability under the null.

    Returns
    -------
    float
        p-value of the combined likelihood-ratio statistic against chi-square
        with two degrees of freedom, or NaN when undefined (fewer than two
        observations, or no violations).
    """
    vH = vHit.astype(int); dE = 1e-12
    if vH.size < 2:
        return np.nan
    iN00 = int(np.sum((vH[:-1] == 0) & (vH[1:] == 0)))
    iN01 = int(np.sum((vH[:-1] == 0) & (vH[1:] == 1)))
    iN10 = int(np.sum((vH[:-1] == 1) & (vH[1:] == 0)))
    iN11 = int(np.sum((vH[:-1] == 1) & (vH[1:] == 1)))
    iX = iN01+iN11; iN = iN00+iN01+iN10+iN11; dPiU = iX/iN
    dLRuc = (-2*(iX*np.log(dP) + (iN - iX)*np.log(1 - dP))
             if iX in (0, iN) else
             -2*((iX*np.log(dP) + (iN - iX)*np.log(1 - dP))
                 - (iX*np.log(dPiU) + (iN - iX)*np.log(1 - dPiU))))
    dPi01 = iN01/max(iN00 + iN01, 1); dPi11 = iN11/max(iN10 + iN11, 1)
    dPi = (iN01 + iN11)/max(iN00 + iN01 + iN10 + iN11, 1)
    if dPi <= 0 or dPi >= 1:
        return np.nan
    dLRind = -2*(((iN00 + iN10)*np.log(1 - dPi + dE) + (iN01 + iN11)*np.log(dPi + dE))
                 - (iN00*np.log(1 - dPi01 + dE) + iN01*np.log(dPi01 + dE)
                    + iN10*np.log(1 - dPi11 + dE) + iN11*np.log(dPi11 + dE)))
    return 1 - chi2.cdf(dLRuc + dLRind, 2)


def tick_loss_vec(vY, vVaR, dP):
    """
    Purpose
    -------
    Asymmetric linear (pinball) loss, strictly consistent for the dP-quantile.

    Parameters
    ----------
    vY : ndarray
        Realised returns (original units).
    vVaR : ndarray
        VaR forecasts at level dP (negative on the loss side).
    dP : float
        Exceedance (tail) probability.

    Returns
    -------
    ndarray
        Per-observation tick loss.
    """
    vE = vY - vVaR
    return (dP - (vE < 0).astype(float))*vE


def fz0_vec(vY, vVaR, vES, dP):
    """
    Purpose
    -------
    FZ0 joint VaR-ES loss (Fissler-Ziegel), strictly consistent for the
    (VaR, ES) pair at level dP.

    Parameters
    ----------
    vY : ndarray
        Realised returns (original units).
    vVaR : ndarray
        VaR forecasts at level dP (negative on the loss side).
    vES : ndarray
        ES forecasts at level dP (negative on the loss side).
    dP : float
        Exceedance (tail) probability.

    Returns
    -------
    ndarray
        Per-observation FZ0 loss.
    """
    vHit = (vY <= vVaR).astype(float)
    return vVaR/vES - vHit*(vVaR - vY)/(dP*vES) + np.log(-vES) - 1.0


def dm_test(vLa, vLb, iLag=None):
    """
    Purpose
    -------
    Diebold-Mariano test on a loss differential with Bartlett HAC long-run
    variance.

    Parameters
    ----------
    vLa : ndarray
        Loss series of model A.
    vLb : ndarray
        Loss series of model B.
    iLag : int or None
        Bartlett truncation lag; defaults to floor(n**(1/3)).

    Returns
    -------
    tuple[float, float]
        (DM statistic, normal-CDF p-value). A negative statistic indicates model
        A has lower loss than model B. Both NaN when fewer than ten finite
        differentials are available.
    """
    vD = (vLa - vLb); vD = vD[np.isfinite(vD)]; iN = vD.size
    if iN < 10:
        return np.nan, np.nan
    if iLag is None:
        iLag = int(np.floor(iN**(1/3)))
    dBar = vD.mean(); vDc = vD - dBar; dVar = np.mean(vDc**2)
    for l in range(1, iLag + 1):
        dVar += 2*(1 - l/(iLag + 1))*np.mean(vDc[l:]*vDc[:-l])
    dDM = dBar/np.sqrt(max(dVar, 1e-12)/iN)
    return dDM, norm.cdf(dDM)


# ===========================================================================
# 4.  Parametric innovation model (Student-t MLE) and VaR/ES multipliers
# ===========================================================================
def student_t_fit(vZ, bFitLoc=False):
    """
    Purpose
    -------
    Fit a location-scale Student-t to the standardised residuals by maximum
    likelihood. Returns NaN parameters (instead of raising) when the finite
    sample is too small, so one degenerate model cannot abort the pipeline.

    Parameters
    ----------
    vZ : ndarray
        Standardised residuals, eps_t = y_t / sqrt(h_{t|t-1}).
    bFitLoc : bool
        If False (default), the location is fixed to 0 (McNeil-Frey mean-zero
        convention) and only (nu, scale) are estimated. If True, (nu, loc, scale)
        are estimated jointly.

    Returns
    -------
    dict
        Student-t parameters and diagnostics: nu (degrees of freedom), loc, scale
        and n_fin (number of finite observations). Parameters are NaN when the
        sample is too small or the fit fails.
    """
    vZ = np.asarray(vZ, dtype=float)
    vZ = vZ[np.isfinite(vZ)]
    if vZ.size < 50:
        return {'nu': np.nan, 'loc': np.nan, 'scale': np.nan, 'n_fin': int(vZ.size)}
    try:
        if bFitLoc:
            dNu, dLoc, dScale = student_t.fit(vZ)
        else:
            dNu, dLoc, dScale = student_t.fit(vZ, floc=0.0)
    except Exception:
        return {'nu': np.nan, 'loc': np.nan, 'scale': np.nan, 'n_fin': int(vZ.size)}
    if not (np.isfinite(dNu) and np.isfinite(dScale) and dNu > 0 and dScale > 0):
        return {'nu': np.nan, 'loc': np.nan, 'scale': np.nan, 'n_fin': int(vZ.size)}
    return {'nu': float(dNu), 'loc': float(dLoc), 'scale': float(dScale),
            'n_fin': int(vZ.size)}


def _student_std_es(dNu, dQ, dP):
    """
    Purpose
    -------
    Lower-tail expected shortfall of the STANDARD Student-t at level dP, given the
    standard-t p-quantile dQ = t_nu^{-1}(dP). Uses the closed form
        ES_std = -(1/p) * (nu + q^2)/(nu - 1) * g_nu(q),
    valid for nu > 1 (finite tail mean).

    Parameters
    ----------
    dNu : float
        Degrees of freedom.
    dQ : float
        Standard-t p-quantile, t_nu^{-1}(dP) (negative for small dP).
    dP : float
        Exceedance (tail) probability.

    Returns
    -------
    float
        Standard-t lower-tail ES (negative), or NaN when nu <= 1.
    """
    if not (np.isfinite(dNu) and dNu > 1.0):
        return np.nan
    dG = student_t.pdf(dQ, dNu)
    return -(1.0/dP)*((dNu + dQ**2)/(dNu - 1.0))*dG


def student_mult(dTail, dP):
    """
    Purpose
    -------
    Parametric Student-t VaR multiplier m and ES multiplier e for tail
    probability dP. Returns (m, e) as positive loss-side multipliers so that
    VaR_t = -m*sqrt(h_t) and ES_t = -e*sqrt(h_t). Returns (nan, nan) on a
    degenerate fit.

    Parameters
    ----------
    dTail : dict
        Student-t fit as returned by student_t_fit, carrying nu, loc, scale.
    dP : float
        Exceedance (tail) probability of the VaR/ES level, e.g. 0.01.

    Returns
    -------
    tuple[float, float]
        (m, e), the standardised-residual VaR and ES multipliers. e is NaN when
        nu <= 1 (undefined tail mean).
    """
    dNu, dLoc, dScale = dTail['nu'], dTail['loc'], dTail['scale']
    if not (np.isfinite(dNu) and np.isfinite(dLoc) and np.isfinite(dScale)
            and dNu > 0 and dScale > 0):
        return np.nan, np.nan
    dQ = student_t.ppf(dP, dNu)                       # standard-t p-quantile (< 0)
    dQeps = dLoc + dScale*dQ                          # residual-scale p-quantile
    dEsStd = _student_std_es(dNu, dQ, dP)             # standard-t ES (< 0) or NaN
    dEsEps = (dLoc + dScale*dEsStd) if np.isfinite(dEsStd) else np.nan
    dM = -dQeps
    dE = -dEsEps if np.isfinite(dEsEps) else np.nan
    return dM, dE


def student_mult_from_row(oRow, dP):
    """
    Purpose
    -------
    Student-t VaR multiplier m from a row of the tails table.

    Parameters
    ----------
    oRow : pandas.Series or dict
        Student-t parameters carrying nu, loc, scale.
    dP : float
        Exceedance (tail) probability of the VaR level.

    Returns
    -------
    float
        VaR multiplier m, or NaN on a degenerate fit.
    """
    dM, _ = student_mult({'nu': oRow['nu'], 'loc': oRow['loc'],
                          'scale': oRow['scale']}, dP)
    return dM


# ===========================================================================
# 5.  Full analysis
# ===========================================================================
def load_or_extend_forecast_cache(vY, iTrain, vGammas, sCachePath, iNJobs=-1):
    """
    Purpose
    -------
    Load a rolling-forecast cache, computing only whichever requested BLADE
    gammas are missing from it rather than the whole grid. 
    Gammas present in the cache but no longer in vGammas are left in the
    file untouched (harmless, just unused by whatever reads vGammas back
    out); nothing is deleted.

    Parameters
    ----------
    vY : ndarray
        Full return series.
    iTrain : int
        Length of the rolling estimation window.
    vGammas : list[float]
        BLADE Barron robustness parameters currently wanted.
    sCachePath : str
        Full path to the forecast cache CSV. Created if it doesn't exist;
        extended in place (new gamma columns computed, merged in, then
        re-saved) if it exists but is missing some of vGammas.
    iNJobs : int
        Number of parallel jobs for the rolling step, used only for
        whichever gammas actually need computing.

    Returns
    -------
    pandas.DataFrame
        The (possibly newly-extended) forecast cache, indexed by step,
        carrying at least every '{model}_hfore' column vGammas and
        BENCH_NAMES need, plus 'scale' and 'y_next'.
    """
    dfCache = None
    if os.path.exists(sCachePath):
        dfCache = pd.read_csv(sCachePath, index_col='step').sort_index()

    lMissingGammas = [g for g in vGammas
                      if dfCache is None or f'BLADE_g{g}_hfore' not in dfCache.columns]

    if not lMissingGammas:
        print(f"All {len(vGammas)} requested gammas already in cache: {sCachePath}")
        return dfCache

    iHave = len(vGammas) - len(lMissingGammas)
    print(f"{iHave} of {len(vGammas)} requested gammas already cached; "
         f"running the rolling step for the other {len(lMissingGammas)}: "
         f"{lMissingGammas}")
    dfNew = rolling_forecasts(vY, iTrain, lMissingGammas, iNJobs)

    if dfCache is None:
        dfCache = dfNew
    else:
        lNewCols = [f'BLADE_g{g}_hfore' for g in lMissingGammas]
        dfCache = dfCache.join(dfNew[lNewCols], how='left')

    os.makedirs(os.path.dirname(sCachePath), exist_ok=True)
    dfCache.to_csv(sCachePath)
    iTotalBlade = sum(1 for c in dfCache.columns
                      if c.startswith('BLADE_g') and c.endswith('_hfore'))
    print(f"Extended cache saved to {sCachePath} ({iTotalBlade} BLADE gammas total).")
    return dfCache


def run_analysis(vY, iTrain=1000, vGammas=None, vTailProb=None,
                 dRefFrac=0.5, bFitLoc=False, iNJobs=-1,
                 sForecastCache=None):
    """
    Purpose
    -------
    Execute the complete parametric BLADE VaR study: rolling BLADE variance
    forecasts, a fixed per-model Student-t innovation fit on a reference block,
    parametric VaR/ES, and backtests with BLADE robustness selection on the
    reference block followed by Diebold-Mariano comparisons of every BLADE model
    against every benchmark on the evaluation block.

    Parameters
    ----------
    vY : ndarray
        Full return series.
    iTrain : int
        Length of the rolling estimation window.
    vGammas : list[float] or None
        BLADE Barron robustness parameters; a default grid is used when None.
    vTailProb : list[float] or None
        Exceedance probabilities p at which VaR/ES is evaluated; a default set is
        used when None.
    dRefFrac : float
        Fraction of the out-of-sample period used as the reference block for the
        Student-t fit and BLADE selection; the remainder is the evaluation block.
    bFitLoc : bool
        Passed to student_t_fit; if False the Student-t location is fixed to 0.
    iNJobs : int
        Number of parallel jobs for the rolling step.
    sForecastCache : str or None
        Filename (not a full path) of the rolling-forecast cache inside the
        results directory. Gammas already present are loaded from it;
        gammas in vGammas that aren't are computed and merged in (see
        load_or_extend_forecast_cache) -- changing vGammas no longer means
        repeating the expensive rolling step for gammas already cached.

    Returns
    -------
    dict
        Keys: 'forecasts' (rolling forecast DataFrame), 'tails' (per-model
        Student-t table), 'backtests' (per-model coverage and loss table), 'dm'
        (BLADE Diebold-Mariano comparisons), 'var_paths' (per-level VaR/ES paths),
        and 'iRef' (reference-block size).
    """
    if vGammas is None:
        vGammas = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
    if vTailProb is None:
        vTailProb = [0.05, 0.025, 0.01]

    # --- one results directory, one consistent cache path for check/read/write
    sCacheBaseDir = "Results_final_param/"
    os.makedirs(sCacheBaseDir, exist_ok=True)
    sCachePath = os.path.join(sCacheBaseDir, sForecastCache) if sForecastCache else \
        os.path.join(sCacheBaseDir, 'blade_param_full_forecasts_btc.csv')

    df = load_or_extend_forecast_cache(vY, iTrain, vGammas, sCachePath, iNJobs)

    lModels = [f'BLADE_g{g}' for g in vGammas] + BENCH_NAMES

    # --- diagnostic: how often does each model produce a usable forecast?
    print("\n=== Forecast diagnostics (fraction NaN over all windows) ===")
    lDead = []
    for sM in lModels:
        if f'{sM}_hfore' not in df.columns:
            print(f"  {sM:<14} MISSING column"); lDead.append(sM); continue
        vH = df[f'{sM}_hfore'].values
        dFracNaN = float(np.mean(~np.isfinite(vH)))
        sFlag = ''
        if dFracNaN > 0.99:
            sFlag = '  <-- failing on (almost) every window'
            lDead.append(sM)
        print(f"  {sM:<14} NaN fraction = {dFracNaN:6.3f}{sFlag}")
    if lDead:
        print(f"\n  NOTE: {lDead} produced no usable forecasts and will be "
              f"dropped from all tables.")

    vScale  = df['scale'].values
    vYn_org = df['y_next'].values*vScale
    iN = len(df); iRef = int(dRefFrac*iN)

    # --- static Student-t innovation fit per model on the reference-block residuals
    dTails, lTail = {}, []
    for sM in lModels:
        if f'{sM}_hfore' not in df.columns:
            continue
        vH = df[f'{sM}_hfore'].values
        vZ = np.where(vH > 0, df['y_next'].values/np.sqrt(np.where(vH > 0, vH, np.nan)), np.nan)
        vZref = vZ[:iRef]; vZref = vZref[np.isfinite(vZref)]
        dT = student_t_fit(vZref, bFitLoc=bFitLoc); dTails[sM] = dT
        lTail.append({'Model': sM, 'nu': dT['nu'], 'loc': dT['loc'],
                      'scale': dT['scale'], 'n_fin': dT['n_fin']})
    dfTails = pd.DataFrame(lTail).set_index('Model')

    print("\n=== Estimated Student-t innovation (MLE on per-model reference residuals) ===")
    print(f"{'Model':<16}{'nu':>8}{'loc':>8}{'scale':>8}{'n_fin':>8}")
    print("-"*48)
    for sM, oR in dfTails.iterrows():
        print(f"  {sM:<14}{oR['nu']:>8.3f}{oR['loc']:>8.4f}{oR['scale']:>8.4f}"
              f"{int(oR['n_fin']):>8d}")

    vYn_eval = vYn_org[iRef:]
    dFZeval   = {dP: {} for dP in vTailProb}
    dTickEval = {dP: {} for dP in vTailProb}
    lBT, lDM = [], []
    dVarPaths = {}

    lLive = list(dTails.keys())

    for dP in vTailProb:
        # full-sample VaR/ES paths for plotting / saving
        mPath = {'y_next_org': vYn_org}
        for sM in lLive:
            vHfull = df[f'{sM}_hfore'].values*vScale**2
            vSigFull = np.sqrt(np.where(vHfull > 0, vHfull, np.nan))
            dM, dE = student_mult(dTails[sM], dP)
            mPath[f'{sM}_VaR'] = -dM*vSigFull
            mPath[f'{sM}_ES']  = -dE*vSigFull
        dVarPaths[dP] = pd.DataFrame(mPath, index=df.index)

        print(f"\n=== VaR/ES backtest, p={dP:.4f} (eval n={iN-iRef}) ===")
        print(f"{'Model':<16}{'mult':>8}{'rate':>8}{'Kup_p':>8}{'CC_p':>8}{'FZ0':>9}{'Tick':>9}")
        print("-"*72)
        for sM in lLive:
            vHorg = df[f'{sM}_hfore'].values*vScale**2
            vSig = np.sqrt(np.where(vHorg > 0, vHorg, np.nan))[iRef:]
            dM, dE = student_mult(dTails[sM], dP)
            if not (np.isfinite(dM) and np.isfinite(dE)):
                continue
            vVaR = -dM*vSig; vES = -dE*vSig
            vMask = np.isfinite(vVaR) & np.isfinite(vES) & np.isfinite(vYn_eval)
            if vMask.sum() == 0:
                continue
            vHit  = (vYn_eval[vMask] < vVaR[vMask]).astype(float)
            vFZ   = fz0_vec(vYn_eval[vMask], vVaR[vMask], vES[vMask], dP)
            vTick = tick_loss_vec(vYn_eval[vMask], vVaR[vMask], dP)
            dFZeval[dP][sM]   = vFZ
            dTickEval[dP][sM] = vTick
            dKupBt = kupiec_uc(vHit, dP)
            dCCBt = christoffersen_cc(vHit, dP)
            print(f"  {sM:<14}{dM:>8.3f}{vHit.mean():>8.4f}"
                  f"{dKupBt:>8.4f}{dCCBt:>8.4f}"
                  f"{np.nanmean(vFZ):>9.4f}{np.nanmean(vTick):>9.4f}")
            lBT.append({'p': dP, 'Model': sM, 'mult': dM,
                        'rate': vHit.mean(), 'Kup_p': dKupBt, 'CC_p': dCCBt,
                        'fz0': np.nanmean(vFZ), 'tick': np.nanmean(vTick)})

        # --- select BLADE gamma on the reference block, then DM on eval block
        dFZref = {}
        for sM in [m for m in lLive if m.startswith('BLADE')]:
            vHref = df[f'{sM}_hfore'].values*vScale**2
            vSig = np.sqrt(np.where(vHref > 0, vHref, np.nan))[:iRef]
            dM, dE = student_mult(dTails[sM], dP)
            if np.isfinite(dM) and np.isfinite(dE):
                vMaskR = np.isfinite(vSig)
                if vMaskR.sum() > 0:
                    dFZref[sM] = np.nanmean(
                        fz0_vec(vYn_org[:iRef][vMaskR], -dM*vSig[vMaskR],
                                -dE*vSig[vMaskR], dP))

        if not dFZref:
            print("  -> no BLADE model produced a finite reference FZ0; skipping selection")
            continue

        sStar = min(dFZref, key=dFZref.get)

        lBladeLive = [m for m in lLive if m.startswith('BLADE') and m in dFZeval[dP]]
        lBenchLive = [b for b in BENCH_NAMES if b in dFZeval[dP]]
        print(f"  -> selected BLADE = {sStar} (reference FZ0={dFZref[sStar]:.4f}); "
              f"DM on eval block (negative => BLADE better), benchmarks: "
              f"{lBenchLive}")

        for sM in lBladeLive:
            sMk = ' *' if sM == sStar else '  '
            dRow = {'p': dP, 'BLADE': sM, 'selected': sM == sStar}
            lParts = []
            for sB in lBenchLive:
                dDM, dPval = dm_test(dFZeval[dP][sM], dFZeval[dP][sB])
                dRow[f'DM_{sB}'] = dDM
                dRow[f'p_{sB}']  = dPval
                lParts.append(f"DM_vs_{sB}={dDM:>7.3f} (p={dPval:.4f})")
            print(f"     {sM:<14}{sMk} " + "  ".join(lParts))
            lDM.append(dRow)

    return {'forecasts': df, 'tails': dfTails,
            'backtests': pd.DataFrame(lBT), 'dm': pd.DataFrame(lDM),
            'var_paths': dVarPaths, 'iRef': iRef}


def double_rolling_window(dOut, iwindow_length):
    """
    Purpose
    ---------
    Regime-visualisation diagnostic (plotting only, not a scored backtest --
    the window ending at t includes t itself; see dynamic_blade_performance, properly-scored version): for every position of a
    sliding window over the evaluation block, which BLADE gamma had the
    lowest mean FZ0 loss in that window.

    Parameters
    ----------
    dOut: dict
        The output dictionary from run_analysis containing 'var_paths', 'iRef', etc.
    iwindow_length : int
        The window length of the second evaluation rolling window.

    Returns
    -------
    dict of pandas.DataFrame
        p -> DataFrame indexed by 'end' step, single column 'best_blade_gamma'.
    """
    iRef = dOut['iRef']
    dRollingResults = {}

    for dP, dfP in dOut['var_paths'].items():
        lBladeModels = [c[:-4] for c in dfP.columns
                        if c.endswith('_VaR') and c.startswith('BLADE_g')]
        dfEval = dfP.iloc[iRef:].copy()
        iN = len(dfEval)

        if iN <= iwindow_length or not lBladeModels:
            print(f"Warning: evaluation block ({iN}) too short, or no live BLADE "
                  f"models, for iwindow_length={iwindow_length} at p={dP}. Skipping.")
            continue

        vY_full = dfEval['y_next_org'].values
        dRollFZ0 = {}
        for sM in lBladeModels:
            vVaR = dfEval[f'{sM}_VaR'].values
            vES = dfEval[f'{sM}_ES'].values
            vFZ0 = fz0_vec(vY_full, vVaR, vES, dP)
            dRollFZ0[sM] = pd.Series(vFZ0, index=dfEval.index).rolling(iwindow_length).mean()

        dfFZ0 = pd.DataFrame(dRollFZ0)
        vBestBlade = dfFZ0.idxmin(axis=1)
        vBestGamma = vBestBlade.apply(
            lambda sM: float(sM.replace('BLADE_g', '')) if isinstance(sM, str) else np.nan)

        dfRolling = pd.DataFrame({'best_blade_gamma': vBestGamma}).iloc[iwindow_length - 1:]
        dRollingResults[dP] = dfRolling

    return dRollingResults


# ===========================================================================
# 6.  Walk-forward "dynamic BLADE" composite and its performance
# ===========================================================================
def dynamic_blade_performance(dOut, iwindow_length=250, vTailProb=None, bVerbose=True):
    """
    Purpose
    -------
    Build a strictly walk-forward "dynamic BLADE" composite and score it.
    Here, at each evaluation date t, the model deployed at t is instead whichever model
    minimised mean FZ0 loss over the trailing iwindow_length evaluation-block
    dates STRICTLY BEFORE t (t itself, and the reference block, are never
    used to pick t's model). 

    Parameters
    ----------
    dOut : dict
        Output of run_analysis; must carry 'var_paths', 'iRef' and 'dm'.
    iwindow_length : int
        Trailing selection-window length, in evaluation-block observations.
    vTailProb : iterable[float] or None
        Tail levels to build the dynamic composite for; defaults to every key
        in dOut['var_paths'].
    bVerbose : bool
        Print the per-p summary and DM lines. Set False when calling this in
        a loop (e.g. a window-length sweep) so the console isn't flooded;
        the full detail is still in the returned DataFrames either way.

    Returns
    -------
    dict
        'series' : dict, p -> pandas.DataFrame indexed by evaluation date from
            iwindow_length onward, columns y_next_org, VaR, ES,
            selected_model, hit, fz0, tick.
        'backtests' : pandas.DataFrame, one row per p, with rate, Kup_p, CC_p,
            fz0, tick, n (one dynamic composite per p, unlike dOut['backtests']
            which has one row per (p, model)).
        'dm' : pandas.DataFrame, one row per (p, comparator), comparator
            ranging over the benchmarks and the statically-selected BLADE
            model for that p ('role' distinguishes the two), with the DM
            statistic and p-value of the dynamic composite's FZ0 loss against
            the comparator's (negative statistic => dynamic composite is
            better). Also carries dynamic_fz0/dynamic_tick/dynamic_rate/
            dynamic_Kup_p/dynamic_CC_p (the composite's own stats, repeated
            per row for convenience) and comparator_fz0/comparator_tick/
            comparator_rate/comparator_Kup_p/comparator_CC_p.
    """
    if vTailProb is None:
        vTailProb = list(dOut['var_paths'].keys())
    iRef = dOut['iRef']
    dfDM = dOut['dm']

    dSeries, lBT, lDM = {}, [], []

    for dP in vTailProb:
        dfP = dOut['var_paths'][dP]
        lModels = [c[:-4] for c in dfP.columns if c.endswith('_VaR')]
        lBladeModels = [m for m in lModels if m.startswith('BLADE')]
        dfEval = dfP.iloc[iRef:].copy()
        vY = dfEval['y_next_org'].values
        iN = len(dfEval)

        if iN <= iwindow_length:
            print(f"Warning: evaluation block ({iN}) too short for "
                  f"iwindow_length={iwindow_length} at p={dP}. Skipping.")
            continue

        # --- per-model FZ0 and tick loss series over the full evaluation block
        # (tick alongside FZ0 so DM rows below can report both loss levels for
        # every comparator, not just the FZ0 differential).
        dFZ0, dTick = {}, {}
        for sM in lModels:
            vVaR = dfEval[f'{sM}_VaR'].values
            vES = dfEval[f'{sM}_ES'].values
            vMask = np.isfinite(vVaR) & np.isfinite(vES) & np.isfinite(vY)
            vFZ = np.full(iN, np.nan)
            vFZ[vMask] = fz0_vec(vY[vMask], vVaR[vMask], vES[vMask], dP)
            dFZ0[sM] = vFZ
            vTk = np.full(iN, np.nan)
            vTk[vMask] = tick_loss_vec(vY[vMask], vVaR[vMask], dP)
            dTick[sM] = vTk

        # --- static BLADE selected for this p, for the DM comparison
        sStatic = None
        if dfDM is not None and not dfDM.empty and 'selected' in dfDM.columns:
            dfSel = dfDM[(dfDM['p'] == dP) & (dfDM['selected'])]
            if not dfSel.empty:
                sStatic = dfSel.iloc[0]['BLADE']

        # --- walk-forward selection and deployment
        lRows, lPos = [], []
        for i in range(iwindow_length, iN):
            dTrail = {sM: np.nanmean(dFZ0[sM][i - iwindow_length:i]) for sM in lBladeModels}
            dTrail = {sM: v for sM, v in dTrail.items() if np.isfinite(v)}
            if not dTrail:
                continue
            sBest = min(dTrail, key=dTrail.get)
            dFZ_i = dFZ0[sBest][i]
            if not np.isfinite(dFZ_i):
                continue
            dVaR_i = dfEval[f'{sBest}_VaR'].values[i]
            dES_i = dfEval[f'{sBest}_ES'].values[i]
            dE_i = vY[i] - dVaR_i
            dTick_i = (dP - float(dE_i < 0))*dE_i
            lPos.append(i)
            lRows.append({'date': dfEval.index[i], 'y_next_org': vY[i],
                         'VaR': dVaR_i, 'ES': dES_i, 'selected_model': sBest,
                         'hit': float(vY[i] < dVaR_i), 'fz0': dFZ_i,
                         'tick': dTick_i})

        if not lRows:
            print(f"  -> no deployable dynamic-composite dates for p={dP}; skipping")
            continue

        dfSeries = pd.DataFrame(lRows).set_index('date')
        dSeries[dP] = dfSeries
        viPos = np.array(lPos)

        vHit = dfSeries['hit'].values
        dKup = kupiec_uc(vHit, dP)
        dCC = christoffersen_cc(vHit, dP)
        if bVerbose:
            print(f"\n=== Dynamic BLADE composite, p={dP:.4f} "
                  f"(walk-forward, trailing {iwindow_length}, n={len(dfSeries)}) ===")
            print(f"  rate={vHit.mean():.4f}  Kup_p={dKup:.4f}  CC_p={dCC:.4f}  "
                  f"FZ0={dfSeries['fz0'].mean():.4f}  Tick={dfSeries['tick'].mean():.4f}")
        lBT.append({'p': dP, 'rate': vHit.mean(), 'Kup_p': dKup, 'CC_p': dCC,
                   'fz0': dfSeries['fz0'].mean(), 'tick': dfSeries['tick'].mean(),
                   'n': len(dfSeries)})

        # --- DM: dynamic composite vs every benchmark, and vs the static BLADE
        lComparators = list(BENCH_NAMES)
        if sStatic is not None:
            lComparators = lComparators + [sStatic]

        dDynFZ0 = dfSeries['fz0'].mean()
        dDynTick = dfSeries['tick'].mean()
        dDynRate = vHit.mean()
        vCompY = vY[viPos]
        for sComp in lComparators:
            if sComp not in dFZ0:
                continue
            vCompAligned = dFZ0[sComp][viPos]
            vCompTickAligned = dTick[sComp][viPos]
            dDM, dPval = dm_test(dfSeries['fz0'].values, vCompAligned)
            dCompFZ0 = float(np.nanmean(vCompAligned))
            dCompTick = float(np.nanmean(vCompTickAligned))

            # Comparator's own coverage stats, on the same iwindow_length-onward
            # dates the composite was scored on (not the whole evaluation
            # block -- see the docstring note on why this can differ from
            # dOut['backtests']).
            vCompVaR = dfEval[f'{sComp}_VaR'].values[viPos]
            vMaskFin = np.isfinite(vCompVaR) & np.isfinite(vCompY)
            vCompHit = (vCompY[vMaskFin] < vCompVaR[vMaskFin]).astype(float)
            dCompRate = float(vCompHit.mean()) if vCompHit.size else np.nan
            dCompKup = kupiec_uc(vCompHit, dP) if vCompHit.size else np.nan
            dCompCC = christoffersen_cc(vCompHit, dP) if vCompHit.size else np.nan

            sRole = 'static_BLADE' if sComp == sStatic else 'benchmark'
            if bVerbose:
                print(f"     vs {sComp:<12} ({sRole}):  DM={dDM:>7.3f} (p={dPval:.4f})  "
                      f"FZ0={dDynFZ0:.4f} vs {dCompFZ0:.4f}  "
                      f"Tick={dDynTick:.4f} vs {dCompTick:.4f}  "
                      f"rate={dDynRate:.4f} vs {dCompRate:.4f}  "
                      f"Kup_p={dKup:.4f} vs {dCompKup:.4f}  "
                      f"CC_p={dCC:.4f} vs {dCompCC:.4f}")
            lDM.append({'p': dP, 'comparator': sComp, 'role': sRole,
                       'DM': dDM, 'p_value': dPval,
                       'dynamic_fz0': dDynFZ0, 'comparator_fz0': dCompFZ0,
                       'dynamic_tick': dDynTick, 'comparator_tick': dCompTick,
                       'dynamic_rate': dDynRate, 'comparator_rate': dCompRate,
                       'dynamic_Kup_p': dKup, 'comparator_Kup_p': dCompKup,
                       'dynamic_CC_p': dCC, 'comparator_CC_p': dCompCC})

    return {'series': dSeries, 'backtests': pd.DataFrame(lBT),
            'dm': pd.DataFrame(lDM)}


# ===========================================================================
# 7.  Full-distribution (log-score) Diebold-Mariano test
# ===========================================================================
# For a model with filtered variance h_t and reference-block Student-t fit
# (nu_hat, mu_hat, s_hat), the implied predictive density of y_t on original
# units is
#
#     f(y_t | F_{t-1}) = (1/sqrt(h_t)) * g_nu( (y_t/sqrt(h_t) - mu_hat)/s_hat ) / s_hat,
#
# with g_nu the standard Student-t density. The per-observation log-score is
#
#     LS_t = log g_nu(z_t) - log(s_hat) - 0.5*log(h_t),   z_t = (y_t/sqrt(h_t) - mu_hat)/s_hat.
#
# The loss series used in the DM test is L_t = -LS_t (lower is better, matching
# the FZ0/tick sign convention used elsewhere in the pipeline).
# ===========================================================================
def log_score_for_model(dfFore, vH_std, oTailRow):
    """
    Purpose
    -------
    Compute the per-observation Student-t log-score for one model over the full
    rolling sample, on original (unstandardised) units.

    Parameters
    ----------
    dfFore : pandas.DataFrame
        Rolling forecast DataFrame (dOut['forecasts']), carrying 'scale' and
        'y_next' (standardised realised next return).
    vH_std : ndarray
        Standardised one-step variance forecast for this model,
        df['{model}_hfore'].values.
    oTailRow : pandas.Series or dict
        Row from dOut['tails'] carrying nu, loc, scale for this model.

    Returns
    -------
    ndarray
        Per-observation log-score LS_t = log g_nu(z_t) - log(s_hat)
        - 0.5*log(h_t), on original units. NaN where h_t <= 0, the fit is
        degenerate, or z_t is not finite.
    """
    dNu, dLoc, dScale = oTailRow['nu'], oTailRow['loc'], oTailRow['scale']
    if not (np.isfinite(dNu) and np.isfinite(dLoc) and np.isfinite(dScale)
            and dNu > 0 and dScale > 0):
        return np.full(len(dfFore), np.nan)

    vScale = dfFore['scale'].values
    vY_org = dfFore['y_next'].values*vScale                 # realised y_t, original units
    vH_org = vH_std*vScale**2                                # h_t, original units

    vSig = np.sqrt(np.where(vH_org > 0, vH_org, np.nan))
    vEps = vY_org/vSig                                        # eps_t = y_t / sqrt(h_t)
    vZ = (vEps - dLoc)/dScale

    vLogG = student_t.logpdf(vZ, dNu)
    vLS = vLogG - np.log(dScale) - 0.5*np.log(np.where(vH_org > 0, vH_org, np.nan))
    return vLS


def log_score_dm_analysis(dOut, lModels=None):
    """
    Purpose
    -------
    Build the per-model log-score loss series (full sample), then run
    Diebold-Mariano of every live BLADE model against every live benchmark on
    the full-distribution log-score loss, evaluation block only. The BLADE
    model marked 'selected' is chosen independently of the FZ0 tail study:
    whichever BLADE model has the lowest mean log-score loss on the
    reference block (the same iRef block each model's Student-t nu/loc/scale
    was fit on).

    Parameters
    ----------
    dOut : dict
        Output dictionary from run_analysis, must carry 'forecasts' and 'tails'
        with the same schema produced there (nu, loc, scale per model).
    lModels : list[str] or None
        Models to include; defaults to every model present in both
        dOut['forecasts'] ('{model}_hfore') and dOut['tails'].index.

    Returns
    -------
    dict
        'scores' : pandas.DataFrame, per-observation log-score loss L_t = -LS_t
            for every live model over the full sample, indexed like
            dOut['forecasts'] (NaN only where a fit is degenerate or h_t <= 0
            -- reference-block rows are real values, not NaN, since they're
            what 'summary_ref' below selects on).
        'summary' : pandas.DataFrame, one row per model with mean log-score
            loss on the evaluation block (lower is better).
        'summary_ref' : pandas.DataFrame, one row per model with mean
            log-score loss on the reference block -- the selection basis for
            'selected' in 'dm' below.
        'dm' : pandas.DataFrame, one row per (BLADE model, benchmark) pair with
            the DM statistic and p-value on the log-score loss differential
            (evaluation block only), negative statistic meaning the BLADE
            model has the better (lower) full-distribution loss. A
            'selected' column marks the BLADE model with the lowest mean
            log-score loss on the reference block.
    """
    df = dOut['forecasts']
    dfTails = dOut['tails']
    iRef = dOut['iRef']

    if lModels is None:
        lModels = [m for m in dfTails.index if f'{m}_hfore' in df.columns]

    dScores = {}
    for sM in lModels:
        vH_std = df[f'{sM}_hfore'].values
        vLS = log_score_for_model(df, vH_std, dfTails.loc[sM])
        dScores[sM] = -vLS                                    # loss: lower is better

    dfScores = pd.DataFrame(dScores, index=df.index)

    dfRef = dfScores.iloc[:iRef]
    dfEval = dfScores.iloc[iRef:]
    lSummary = [{'Model': sM, 'log_score_loss': dfEval[sM].mean(skipna=True),
                'n_fin': int(np.isfinite(dfEval[sM].values).sum())}
               for sM in lModels]
    dfSummary = pd.DataFrame(lSummary).set_index('Model').sort_values('log_score_loss')

    lSummaryRef = [{'Model': sM, 'log_score_loss': dfRef[sM].mean(skipna=True),
                   'n_fin': int(np.isfinite(dfRef[sM].values).sum())}
                  for sM in lModels]
    dfSummaryRef = pd.DataFrame(lSummaryRef).set_index('Model').sort_values('log_score_loss')

    print("\n=== Full-distribution log-score, evaluation block (lower is better) ===")
    print(f"{'Model':<16}{'LogScoreLoss':>14}{'n_fin':>8}")
    print("-"*40)
    for sM, oR in dfSummary.iterrows():
        print(f"  {sM:<14}{oR['log_score_loss']:>14.4f}{int(oR['n_fin']):>8d}")

    lBladeLive = [m for m in lModels if m.startswith('BLADE')]
    lBenchLive = [b for b in BENCH_NAMES if b in lModels]

    # --- select the BLADE model on the reference block using log-score
    # itself so the star marked
    # below is never chosen using the evaluation-block data it's then
    # DM-tested on.
    dfSummaryRefBlade = dfSummaryRef[dfSummaryRef.index.str.startswith('BLADE')]
    dfSummaryRefBlade = dfSummaryRefBlade[np.isfinite(dfSummaryRefBlade['log_score_loss'])]
    sStarLS = dfSummaryRefBlade['log_score_loss'].idxmin() if not dfSummaryRefBlade.empty else None

    # For cross-reference only -- the FZ0 tail study's pick, printed but not
    # used to select or restrict anything here.
    sStarFZ0 = None
    if 'dm' in dOut and not dOut['dm'].empty and 'selected' in dOut['dm'].columns:
        dfSel = dOut['dm'][dOut['dm']['selected']]
        if not dfSel.empty:
            sStarFZ0 = dfSel.iloc[0]['BLADE']

    print(f"\n  -> DM on eval block (negative => BLADE better), benchmarks: {lBenchLive}")
    if sStarLS is not None:
        print(f"  -> selected BLADE (reference-block log-score) = {sStarLS} "
              f"(reference log-score={dfSummaryRefBlade.loc[sStarLS, 'log_score_loss']:.4f})")
    else:
        print("  -> no BLADE model produced a finite reference-block log-score; no star selected")
    if sStarFZ0 is not None:
        print(f"  -> (for comparison) FZ0-selected BLADE at the first tail level was {sStarFZ0}")

    lDM = []
    for sM in lBladeLive:
        dRow = {'BLADE': sM, 'selected': sM == sStarLS}
        lParts = []
        for sB in lBenchLive:
            dDM, dPval = dm_test(dfEval[sM].values, dfEval[sB].values)
            dRow[f'DM_{sB}'] = dDM
            dRow[f'p_{sB}'] = dPval
            lParts.append(f"DM_vs_{sB}={dDM:>7.3f} (p={dPval:.4f})")
        sMk = ' *' if sM == sStarLS else '  '
        print(f"     {sM:<14}{sMk} " + "  ".join(lParts))
        lDM.append(dRow)

    return {'scores': dfScores, 'summary': dfSummary, 'summary_ref': dfSummaryRef,
            'dm': pd.DataFrame(lDM)}


def dynamic_blade_performance_logscore(dOut, dLogScore, iwindow_length=250, bVerbose=True):
    """
    Purpose
    -------
    At each evaluation date t,
    the model deployed at t is whichever model had the lowest mean
    full-distribution log-score loss over the trailing iwindow_length
    evaluation-block dates STRICTLY BEFORE t (t itself is never used to pick
    its own model). Scored on the same log-score loss it was selected on.

    Parameters
    ----------
    dOut : dict
        Output of run_analysis; must carry 'iRef'.
    dLogScore : dict
        Output of log_score_dm_analysis(dOut); must carry 'scores' (per-model
        log-score loss series, full sample) and 'dm' (carries the
        reference-block log-score-selected static BLADE model in its
        'selected' column, used for the DM comparison here).
    iwindow_length : int
        Trailing selection-window length, in evaluation-block observations.
        Matches the window used by dynamic_blade_performance.
    bVerbose : bool
        Print the summary and DM lines. Set False when calling this in a
        loop (e.g. a window-length sweep) so the console isn't flooded; the
        full detail is still in the returned DataFrames either way.

    Returns
    -------
    dict
        'series' : pandas.DataFrame indexed by evaluation date from
            iwindow_length onward, columns selected_model, log_score_loss.
        'summary' : dict, mean log-score loss and n over the deployed dates.
        'dm' : pandas.DataFrame, one row per comparator (the benchmarks and
            the reference-block log-score-optimal static BLADE model, 'role'
            distinguishing the two), with the DM statistic and p-value of the
            dynamic composite's log-score loss against the comparator's
            (negative statistic => dynamic composite is better). Also
            carries dynamic_log_score (the composite's own mean log-score
            loss, repeated per row for convenience) and
            comparator_log_score (the comparator's mean log-score loss over
            the same dates), so this table alone is enough to build a full
            comparison table without joining back to 'summary'.
    """
    iRef = dOut['iRef']
    dfScoresEval = dLogScore['scores'].iloc[iRef:]
    lModels = list(dfScoresEval.columns)
    lBladeModels = [m for m in lModels if m.startswith('BLADE')]
    iN = len(dfScoresEval)

    if iN <= iwindow_length:
        print(f"Warning: evaluation block ({iN}) too short for "
              f"iwindow_length={iwindow_length}. Skipping.")
        return {'series': pd.DataFrame(), 'summary': {}, 'dm': pd.DataFrame()}

    # --- reference-block log-score-optimal static BLADE, for the DM comparison.
    # Read off log_score_dm_analysis's 'selected' column (reference-block
    # log-score argmin) rather than dLogScore['summary'] (evaluation-block
    # mean): selecting on evaluation-block performance and then DM-testing
    # against a subset of that same evaluation block would be circular.
    dfDM_ls = dLogScore['dm']
    sStatic = None
    if dfDM_ls is not None and not dfDM_ls.empty and 'selected' in dfDM_ls.columns:
        dfSel = dfDM_ls[dfDM_ls['selected']]
        if not dfSel.empty:
            sStatic = dfSel.iloc[0]['BLADE']

    # --- walk-forward selection and deployment
    lRows, lPos = [], []
    for i in range(iwindow_length, iN):
        dTrail = {sM: np.nanmean(dfScoresEval[sM].values[i - iwindow_length:i])
                 for sM in lBladeModels}
        dTrail = {sM: v for sM, v in dTrail.items() if np.isfinite(v)}
        if not dTrail:
            continue
        sBest = min(dTrail, key=dTrail.get)
        dLoss_i = dfScoresEval[sBest].values[i]
        if not np.isfinite(dLoss_i):
            continue
        lPos.append(i)
        lRows.append({'date': dfScoresEval.index[i], 'selected_model': sBest,
                     'log_score_loss': dLoss_i})

    if not lRows:
        print("  -> no deployable dynamic-composite dates for log-score; skipping")
        return {'series': pd.DataFrame(), 'summary': {}, 'dm': pd.DataFrame()}

    dfSeries = pd.DataFrame(lRows).set_index('date')
    viPos = np.array(lPos)

    dSummary = {'log_score_loss': dfSeries['log_score_loss'].mean(),
               'n': len(dfSeries)}
    if bVerbose:
        print(f"\n=== Dynamic BLADE composite, full-distribution log-score "
              f"(walk-forward, trailing {iwindow_length}, n={len(dfSeries)}) ===")
        print(f"  LogScoreLoss={dSummary['log_score_loss']:.4f}")

    # --- DM: dynamic composite vs every benchmark, and vs the static BLADE
    lComparators = list(BENCH_NAMES)
    if sStatic is not None:
        lComparators = lComparators + [sStatic]

    dDynLogScore = dSummary['log_score_loss']
    lDM = []
    for sComp in lComparators:
        if sComp not in dfScoresEval.columns:
            continue
        vCompAligned = dfScoresEval[sComp].values[viPos]
        dDM, dPval = dm_test(dfSeries['log_score_loss'].values, vCompAligned)
        dCompLogScore = float(np.nanmean(vCompAligned))
        sRole = 'static_BLADE' if sComp == sStatic else 'benchmark'
        if bVerbose:
            print(f"     vs {sComp:<12} ({sRole}):  DM={dDM:>7.3f} (p={dPval:.4f})  "
                  f"LogScore={dDynLogScore:.4f} vs {dCompLogScore:.4f}")
        lDM.append({'comparator': sComp, 'role': sRole, 'DM': dDM, 'p_value': dPval,
                   'dynamic_log_score': dDynLogScore, 'comparator_log_score': dCompLogScore})

    return {'series': dfSeries, 'summary': dSummary, 'dm': pd.DataFrame(lDM)}


def double_rolling_window_logscore(dOut, dLogScore, iwindow_length):
    """
    Purpose
    -------
    Full-distribution (log-score) counterpart of double_rolling_window: for
    the window ending at date t (t included -- regime-visualisation only,
    not a scored walk-forward composite; see
    dynamic_blade_performance_logscore for the causal, properly-scored
    version), which BLADE model had the lowest mean log-score loss over that
    window.

    Parameters
    ----------
    dOut : dict
        Output of run_analysis; needs 'iRef'.
    dLogScore : dict
        Output of log_score_dm_analysis(dOut); needs 'scores'.
    iwindow_length : int
        The window length of the second evaluation rolling window.

    Returns
    -------
    pandas.DataFrame
        Indexed by 'end' step, single column 'best_blade_gamma'.
    """
    iRef = dOut['iRef']
    dfScoresEval = dLogScore['scores'].iloc[iRef:]
    lBladeModels = [c for c in dfScoresEval.columns if c.startswith('BLADE_g')]
    iTotalRows = len(dfScoresEval)

    if iTotalRows <= iwindow_length or not lBladeModels:
        print(f"Warning: evaluation block ({iTotalRows}) too short, or no live BLADE "
              f"models, for iwindow_length={iwindow_length}. Skipping.")
        return pd.DataFrame()

    dfRollLS = dfScoresEval[lBladeModels].rolling(iwindow_length).mean()
    vBestBlade = dfRollLS.idxmin(axis=1)
    vBestGamma = vBestBlade.apply(
        lambda sM: float(sM.replace('BLADE_g', '')) if isinstance(sM, str) else np.nan)

    return pd.DataFrame({'best_blade_gamma': vBestGamma}).iloc[iwindow_length - 1:]


def dynamic_window_sweep(dOut, dLogScore,
                         lWindows=(100, 150, 200, 250, 350, 500, 750, 1000, 1250),
                         bVerbose=True):
    """
    Purpose
    -------
    Run both dynamic composites (FZ0-based and log-score-based) across a grid
    of trailing selection-window lengths on the same cached forecasts.

    Parameters
    ----------
    dOut : dict
        Output of run_analysis.
    dLogScore : dict
        Output of log_score_dm_analysis(dOut).
    lWindows : iterable[int]
        Trailing selection-window lengths to sweep.
    bVerbose : bool
        Print the dynamic-composite-vs-benchmarks DM test for every window
        length, in the same per-window format dynamic_blade_performance /
        dynamic_blade_performance_logscore use on their own (one block per
        window, one line per benchmark). The static-BLADE comparator is
        computed and still returned in the DataFrames, but not printed here
        -- see the "forget about the fixed gamma comparison" discussion: it
        doesn't tell you what to do prospectively, so it isn't part of the
        headline console story. Full detail is in the returned DataFrames
        either way.

    Returns
    -------
    dict
        'fz0_backtests' : pandas.DataFrame, one row per (window, p): n,
            switches, switch_pct, distinct_models, rate, Kup_p, CC_p, fz0,
            tick.
        'fz0_dm' : pandas.DataFrame, one row per (window, p, comparator):
            role, DM, p_value, dynamic_fz0, comparator_fz0, dynamic_tick,
            comparator_tick (see dynamic_blade_performance's 'dm').
        'logscore_backtests' : pandas.DataFrame, one row per window: n,
            switches, switch_pct, distinct_models, log_score_loss.
        'logscore_dm' : pandas.DataFrame, one row per (window, comparator):
            role, DM, p_value, dynamic_log_score, comparator_log_score (see
            dynamic_blade_performance_logscore's 'dm').
    """
    lFZ0BT, lFZ0DM, lLSBT, lLSDM = [], [], [], []

    print(f"Window-length sweep over {list(lWindows)}...")
    for iW in lWindows:
        if not bVerbose:
            print(f"  window={iW} ...", end=' ', flush=True)

        dDyn = dynamic_blade_performance(dOut, iwindow_length=iW, bVerbose=False)
        for dP, dfSeries in dDyn['series'].items():
            iN = len(dfSeries)
            iSw = int((dfSeries['selected_model'] != dfSeries['selected_model'].shift()).sum()) - 1
            iUniq = int(dfSeries['selected_model'].nunique())
            dKup = kupiec_uc(dfSeries['hit'].values, dP)
            dCC = christoffersen_cc(dfSeries['hit'].values, dP)
            lFZ0BT.append({'window': iW, 'p': dP, 'n': iN, 'switches': iSw,
                          'switch_pct': 100*iSw/iN, 'distinct_models': iUniq,
                          'rate': dfSeries['hit'].mean(), 'Kup_p': dKup, 'CC_p': dCC,
                          'fz0': dfSeries['fz0'].mean(), 'tick': dfSeries['tick'].mean()})

            if bVerbose:
                print(f"\n=== Dynamic BLADE composite, p={dP:.4f} "
                      f"(walk-forward, trailing {iW}, n={iN}) ===")
                print(f"  rate={dfSeries['hit'].mean():.4f}  Kup_p={dKup:.4f}  CC_p={dCC:.4f}  "
                      f"FZ0={dfSeries['fz0'].mean():.4f}  Tick={dfSeries['tick'].mean():.4f}")
                dfBenchDM = dDyn['dm'][(dDyn['dm']['p'] == dP)
                                      & (dDyn['dm']['role'] == 'benchmark')]
                for _, oRow in dfBenchDM.iterrows():
                    print(f"     vs {oRow['comparator']:<12} (benchmark):  "
                          f"DM={oRow['DM']:>7.3f} (p={oRow['p_value']:.4f})  "
                          f"FZ0={oRow['comparator_fz0']:.4f}  Tick={oRow['comparator_tick']:.4f}  "
                          f"rate={oRow['comparator_rate']:.4f}  "
                          f"Kup_p={oRow['comparator_Kup_p']:.4f}  "
                          f"CC_p={oRow['comparator_CC_p']:.4f}")

        if not dDyn['dm'].empty:
            dfDM = dDyn['dm'].copy()
            dfDM.insert(0, 'window', iW)
            lFZ0DM.append(dfDM)

        dDynLS = dynamic_blade_performance_logscore(dOut, dLogScore, iwindow_length=iW,
                                                    bVerbose=False)
        if not dDynLS['series'].empty:
            dfS = dDynLS['series']
            iN = len(dfS)
            iSw = int((dfS['selected_model'] != dfS['selected_model'].shift()).sum()) - 1
            iUniq = int(dfS['selected_model'].nunique())
            lLSBT.append({'window': iW, 'n': iN, 'switches': iSw,
                         'switch_pct': 100*iSw/iN, 'distinct_models': iUniq,
                         'log_score_loss': dfS['log_score_loss'].mean()})

            if bVerbose:
                print(f"\n=== Dynamic BLADE composite, full-distribution log-score "
                      f"(walk-forward, trailing {iW}, n={iN}) ===")
                print(f"  LogScoreLoss={dfS['log_score_loss'].mean():.4f}")
                dfBenchDM = dDynLS['dm'][dDynLS['dm']['role'] == 'benchmark']
                for _, oRow in dfBenchDM.iterrows():
                    print(f"     vs {oRow['comparator']:<12} (benchmark):  "
                          f"DM={oRow['DM']:>7.3f} (p={oRow['p_value']:.4f})  "
                          f"LogScore={oRow['comparator_log_score']:.4f}")

        if not dDynLS['dm'].empty:
            dfDM = dDynLS['dm'].copy()
            dfDM.insert(0, 'window', iW)
            lLSDM.append(dfDM)

        if not bVerbose:
            print("done")

    dfFZ0BT = pd.DataFrame(lFZ0BT)
    dfFZ0DM = pd.concat(lFZ0DM, ignore_index=True) if lFZ0DM else pd.DataFrame()
    dfLSBT = pd.DataFrame(lLSBT)
    dfLSDM = pd.concat(lLSDM, ignore_index=True) if lLSDM else pd.DataFrame()

    return {'fz0_backtests': dfFZ0BT, 'fz0_dm': dfFZ0DM,
            'logscore_backtests': dfLSBT, 'logscore_dm': dfLSDM}

