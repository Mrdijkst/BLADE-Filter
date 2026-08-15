#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots: plotting helpers for the parametric BLADE Value-at-Risk analysis.

Author: Mathijs Dijkstra
Date:   August 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Basis import student_mult_from_row

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False,
})


def plot_param_var_paths(dfFore, dfTails, dP=0.01, lModels=None,
                         iStart=None, iEnd=None, sTitle=None, sSavePath=None):
    """
    Purpose
    -------
    Plot the parametric Student-t VaR paths exactly as run_analysis
    constructs them.

    Parameters
    ----------
    dfFore : pandas.DataFrame
        Rolling forecast DataFrame from run_analysis.
    dfTails : pandas.DataFrame
        Per-model Student-t table from run_analysis.
    dP : float
        Exceedance (tail) probability of the plotted VaR level.
    lModels : list[str] or None
        Models to plot; defaults to every model with a forecast column and a fit.
    iStart, iEnd : int or None
        Optional slice bounds on the rolling index for zooming.
    sTitle : str or None
        Plot title.
    sSavePath : str or None
        If given, the figure is written to this path.

    Returns
    -------
    pandas.DataFrame
        The plotted returns and per-model VaR paths.
    """
    d = dfFore.iloc[iStart:iEnd]
    vScale = d['scale'].values
    vY_org = d['y_next'].values*vScale

    if lModels is None:
        lModels = [c[:-6] for c in dfFore.columns if c.endswith('_hfore')]
    lModels = [m for m in lModels if m in dfTails.index]

    mOut = {'y_next_org': vY_org}
    plt.figure(figsize=(12, 6))
    plt.plot(d.index, vY_org, color='black', lw=0.8, label='Return')
    for sM in lModels:
        dM = student_mult_from_row(dfTails.loc[sM], dP)
        if not np.isfinite(dM):
            continue
        vH = d[f'{sM}_hfore'].values
        vSig = vScale*np.sqrt(np.where(vH > 0, vH, np.nan))
        vVaR = -dM*vSig
        mOut[f'{sM}_VaR'] = vVaR
        plt.plot(d.index, vVaR, lw=0.7, label=sM)
    plt.title(sTitle or f'{100*dP:.1f}% Student-t VaR forecasts')
    plt.legend(fontsize=8)
    plt.tight_layout()
    if sSavePath:
        plt.savefig(sSavePath, dpi=150)
    plt.show()
    return pd.DataFrame(mOut, index=d.index)


def plot_param_multipliers(dfTails, vTailProb=(0.05, 0.025, 0.01, 0.005), sSavePath=None):
    """
    Purpose
    -------
    Bar chart of the per-model Student-t VaR multiplier m(model, p) across levels.

    Parameters
    ----------
    dfTails : pandas.DataFrame
        Per-model Student-t table from run_analysis.
    vTailProb : iterable[float]
        Exceedance probabilities p at which the multiplier is plotted.
    sSavePath : str or None
        If given, the figure is written to this path.

    Returns
    -------
    pandas.DataFrame
        Per-model multipliers, models on the index and tail probabilities on the
        columns.
    """
    lModels = list(dfTails.index)
    mM = {dP: [student_mult_from_row(dfTails.loc[sM], dP) for sM in lModels]
          for dP in vTailProb}
    dfM = pd.DataFrame(mM, index=lModels)

    iG = len(vTailProb)
    x = np.arange(len(lModels)); w = 0.8/iG
    plt.figure(figsize=(12, 5))
    for j, dP in enumerate(vTailProb):
        plt.bar(x + j*w, dfM[dP].values, width=w, label=f'p={dP}')
    plt.xticks(x + 0.4 - w/2, lModels, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Student-t multiplier  m'); plt.title('Per-model Student-t VaR multipliers')
    plt.legend(); plt.tight_layout()
    if sSavePath:
        plt.savefig(sSavePath, dpi=150)
    plt.show()
    return dfM


def plot_regime_grid(dRollingByP, dfRollingLS, dOut, iwindow_length, sSavePath=None,
                     sStyle='dots', tXlim=None):
    """
    Purpose
    -------
    Combine the FZ0 regime plots (one per tail level) and the log-score
    regime plot into a single figure sharing one x-axis: one realised-return
    panel on top, then one gamma-regime panel per criterion below it, so the
    same stretch of history can be compared across criteria directly instead
    of flipping between separate images. 

    Parameters
    ----------
    dRollingByP : dict
        p -> pandas.DataFrame, output of double_rolling_window for that p,
        all at the same iwindow_length as dfRollingLS.
    dfRollingLS : pandas.DataFrame
        Output of double_rolling_window_logscore, same iwindow_length.
    dOut : dict
        Output of run_analysis; used for the realised-return panel.
    iwindow_length : int
        The window length these were computed at (title only).
    sSavePath : str or None
        If given, the figure is written to this path.
    sStyle : str
        'dots' (default, original behaviour): scatter markers only, no
        connecting line. 'lines': connecting line only, no markers --
        makes the staircase/plateau structure of the best-gamma track
        visually explicit. 'both': line with markers, useful when zoomed
        in enough that individual switch points are still legible.
    tXlim : tuple[float, float] or None
        If given, (xmin, xmax) applied to every panel's x-axis -- zooms
        the figure to that step range without recomputing anything (all
        panels still hold the full series underneath).

    Returns
    -------
    None
    """
    lPanels = list(dRollingByP.items()) + [('logscore', dfRollingLS)]
    lPanels = [(k, v) for k, v in lPanels if v is not None and not v.empty]
    if not lPanels:
        print(f"No rolling data to plot for window={iwindow_length}.")
        return

    iRows = 1 + len(lPanels)
    fig, axes = plt.subplots(iRows, 1, figsize=(13, 2.6*iRows), sharex=True)
    axRet = axes[0]

    sFirstKey, dfFirst = lPanels[0]
    if sFirstKey == 'logscore':
        dfFore = dOut['forecasts']
        vY = (dfFore['y_next']*dfFore['scale']).loc[dfFirst.index]
    else:
        vY = dOut['var_paths'][sFirstKey]['y_next_org'].loc[dfFirst.index]

    axRet.plot(dfFirst.index, vY, color='black', lw=0.8, label='Realized Return')
    axRet.axhline(0, color='gray', lw=0.5, ls='--')
    axRet.set_ylabel('Returns', fontsize=9)
    sStyleSuffix = {'dots': '', 'lines': ' (lines)', 'both': ' (dots+lines)'}[sStyle]
    sZoomSuffix = f' [zoomed {int(tXlim[0])}-{int(tXlim[1])}]' if tXlim is not None else ''
    # axRet.set_title(rf'Market Regimes vs. Dynamic Optimal Robustness ($\gamma$) | '
    #                 f'window={iwindow_length}{sStyleSuffix}{sZoomSuffix}',
    #                 fontsize=13, fontweight='bold')
    axRet.grid(True, alpha=0.25)
    axRet.legend(loc='upper left', fontsize=8)

    for ax, (sKey, dfRolling) in zip(axes[1:], lPanels):
        vBladeGamma = dfRolling['best_blade_gamma']

        if sStyle == 'dots':
            ax.scatter(dfRolling.index, vBladeGamma, color='royalblue', s=10, alpha=0.8,
                      edgecolors='none', label=r'Best BLADE $\gamma$')
        elif sStyle == 'lines':
            # Plain point-to-point line, same style as the Realized Return
            # panel above (no markers, default straight-segment drawstyle).
            ax.plot(dfRolling.index, vBladeGamma, lw=1.1, color='royalblue',
                   label=r'Best BLADE $\gamma$')
        else:  # 'both'
            ax.plot(dfRolling.index, vBladeGamma, lw=1.1, color='royalblue',
                   marker='o', ms=3.5, markerfacecolor='royalblue', markeredgecolor='none',
                   label=r'Best BLADE $\gamma$')

        sLabel = f'p={sKey}' if sKey != 'logscore' else 'log-score'
        ax.set_ylabel(rf'$\gamma$ ({sLabel})', fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper left', fontsize=7)

    axes[-1].set_xlabel('Evaluation Step Index', fontsize=10)
    if tXlim is not None:
        for ax in axes:
            ax.set_xlim(tXlim)
    plt.tight_layout()
    if sSavePath:
        plt.savefig(sSavePath, dpi=150)
        print(f"Saved combined regime grid to {sSavePath}")
    plt.show()

