#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RiskMain: entry point for the parametric BLADE Value-at-Risk analysis.

Author: Mathijs Dijkstra
Date:   August 2026
"""

import os
import numpy as np
import pandas as pd

from Basis import (run_analysis,
                       double_rolling_window,
                       double_rolling_window_logscore,
                       dynamic_window_sweep,
                       log_score_dm_analysis)
from Plots import (plot_param_var_paths,
                       plot_param_multipliers,
                       plot_regime_grid)


def main():
    """
    Purpose
    -------
    Run the full parametric BLADE VaR study on the Bitcoin return series and write
    all tables and figures to the results directory.

    Returns
    -------
    dict
        The run_analysis output dictionary (forecasts, tails, backtests, dm,
        var_paths, iRef).
    """
    dfData = pd.read_csv('blade_returns_Bitcoin.csv')
    vY = np.asarray(dfData['ret'], dtype=float)

    # Fine 0.01-spacing gamma grid, 2.05 down to 0.8 (126 values).
    vGammas = [2.05, 2.04, 2.03, 2.02, 2.01,
    2.00, 1.99, 1.98, 1.97, 1.96, 1.95, 1.94, 1.93, 1.92, 1.91, 
    1.90, 1.89, 1.88, 1.87, 1.86, 1.85, 1.84, 1.83, 1.82, 1.81,
    1.80, 1.79, 1.78, 1.77, 1.76, 1.75, 1.74, 1.73, 1.72, 1.71,
    1.70, 1.69, 1.68, 1.67, 1.66, 1.65, 1.64, 1.63, 1.62, 1.61,
    1.60, 1.59, 1.58, 1.57, 1.56, 1.55, 1.54, 1.53, 1.52, 1.51,
    1.50, 1.49, 1.48, 1.47, 1.46, 1.45, 1.44, 1.43, 1.42, 1.41,
    1.40, 1.39, 1.38, 1.37, 1.36, 1.35, 1.34, 1.33, 1.32, 1.31,
    1.30, 1.29, 1.28, 1.27, 1.26, 1.25, 1.24, 1.23, 1.22, 1.21,
    1.20, 1.19, 1.18, 1.17, 1.16, 1.15, 1.14, 1.13, 1.12, 1.11,
    1.10, 1.09, 1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.02, 1.01,
    1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91,
    0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81,
    0.80]


              
    vTailProb = [0.05, 0.025, 0.01]

    # Fraction of the out-of-sample period used as the reference block (tail
    # fit + static gamma selection); the rest is the evaluation block.
    dRefFrac = 0.3

    dOut = run_analysis(vY, iTrain=700,
                        vGammas=vGammas, vTailProb=vTailProb,
                        dRefFrac=dRefFrac, bFitLoc=False, iNJobs=-1,
                        sForecastCache='blade_param_xi06_forecasts_btc.csv')

    out_base_dir = f"Results_final_param_xi06_frac{dRefFrac}/"
    os.makedirs(out_base_dir, exist_ok=True)

    dOut['forecasts'].to_csv(f'{out_base_dir}blade_param_xi06_forecasts_btc.csv')
    dOut['tails'].to_csv(f'{out_base_dir}blade_param_xi06_studentt_btc.csv')
    dOut['backtests'].to_csv(f'{out_base_dir}blade_param_xi06_backtest_btc.csv')
    dOut['dm'].to_csv(f'{out_base_dir}blade_param_xi06_dm_btc.csv')

    for dP, dfP in dOut['var_paths'].items():
        dfP.to_csv(f'{out_base_dir}blade_param_xi06_varpaths_q{dP}_btc.csv')

    print("\nSaved forecasts, Student-t fits, backtest, DM and VaR-path CSVs.")

    plot_param_var_paths(dOut['forecasts'], dOut['tails'], dP=0.01,
                         sTitle='1.0% Student-t VaR forecasts',
                         sSavePath=f'{out_base_dir}blade_param_xi06_var_q01_btc.png')
    plot_param_var_paths(dOut['forecasts'], dOut['tails'], dP=0.01,
                         iStart=900, iEnd=1000,
                         sTitle='1.0% Student-t VaR forecasts (zoomed)',
                         sSavePath=f'{out_base_dir}blade_param_xi06_var_q01_zoom_btc.png')
    plot_param_multipliers(dOut['tails'], vTailProb=vTailProb,
                           sSavePath=f'{out_base_dir}blade_param_xi06_multipliers_btc.png')

    dLS = log_score_dm_analysis(dOut)
    dLS['summary'].to_csv(f'{out_base_dir}blade_param_xi06_logscore_summary.csv')
    dLS['dm'].to_csv(f'{out_base_dir}blade_param_xi06_logscore_dm.csv')

    # =========================================================================
    # Second-Stage Rolling Window Evaluation & Visualizations
    # =========================================================================
    # Regime-visualisation diagnostics only (not a scored backtest -- the
    # window ending at t includes t itself; see dynamic_window_sweep below
    # for the properly causal, scored version). 
    for iEvalWindow in (100, 250, 500):
        print(f"\nRunning second-stage rolling window evaluation (window={iEvalWindow})...")

        dRollingResults = double_rolling_window(dOut, iwindow_length=iEvalWindow)

        for dP, dfRolling in dRollingResults.items():
            sCsvPath = f'{out_base_dir}blade_param_xi06_rolling_eval_q{dP}_w{iEvalWindow}.csv'
            dfRolling.to_csv(sCsvPath)
            print(f"Saved rolling evaluation summary for p={dP} to {sCsvPath}")

        dfRollingLS = double_rolling_window_logscore(dOut, dLS, iwindow_length=iEvalWindow)
        dfRollingLS.to_csv(f'{out_base_dir}blade_param_xi06_rolling_eval_logscore_w{iEvalWindow}.csv')

        plot_regime_grid(
            dRollingResults, dfRollingLS, dOut, iEvalWindow,
            sSavePath=f'{out_base_dir}blade_param_xi06_regime_grid_w{iEvalWindow}.pdf')

        # Same data, two extra views: a lines-only full-range version, and a zoomed-in dots+lines version.
        plot_regime_grid(
            dRollingResults, dfRollingLS, dOut, iEvalWindow, sStyle='lines',
            sSavePath=f'{out_base_dir}blade_param_xi06_regime_grid_lines_w{iEvalWindow}.pdf')

        iZoomStart = int(min(dfR.index.min() for dfR in dRollingResults.values() if not dfR.empty))
        plot_regime_grid(
            dRollingResults, dfRollingLS, dOut, iEvalWindow, sStyle='both',
            tXlim=(iZoomStart, iZoomStart + 650),
            sSavePath=f'{out_base_dir}blade_param_xi06_regime_grid_zoom_w{iEvalWindow}.pdf')

    # =========================================================================
    # Walk-forward dynamic-BLADE composite
    # =========================================================================

    print("\nRunning window-length sweep for the dynamic BLADE composite...")

    dSweep = dynamic_window_sweep(dOut, dLS,
                                  lWindows=(100, 150, 200, 250, 350, 500))

    dSweep['fz0_backtests'].to_csv(f'{out_base_dir}blade_param_xi06_dynamic_windowsweep_fz0_backtest.csv',
                                   index=False)
    dSweep['fz0_dm'].to_csv(f'{out_base_dir}blade_param_xi06_dynamic_windowsweep_fz0_dm.csv',
                            index=False)
    dSweep['logscore_backtests'].to_csv(
        f'{out_base_dir}blade_param_xi06_dynamic_windowsweep_logscore_backtest.csv', index=False)
    dSweep['logscore_dm'].to_csv(
        f'{out_base_dir}blade_param_xi06_dynamic_windowsweep_logscore_dm.csv', index=False)
    print(f"Saved window-sweep backtest and DM CSVs to {out_base_dir}")

    return dOut


if __name__ == '__main__':
    main()
