#!/usr/bin/env python3
"""
Basis functions 
"""

import numpy as np
#from LocationModels import RobustQLEModel, OracleStudentTLocation
#import matplotlib.pyplot as plt
from scipy.special import gammaln
#import numpy as np

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

def nu_tilde_for_10x_sd(nu):
    R = 100 * (nu / (nu - 2))
    return 2 * R / (R - 1)

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
###########################################################
def MPITaskDistributor(iRank, iProc, iTotal, bOrder=True):
    """
    Purpose
    ----------
    Construct vector of indices [integers] for which process with rank iRank should do calculations

    Parameters
    ----------
    iRank :     integer, rank running process
    iProc :     integer, total number of processes in MPI program
    iTotal :    integer, total number of tasks
    bOrder :    boolean, use standard ordering of integers if True

    Output
    ----------
    vInt :      vector, part of total indices 0 to [not incl.] iTotal that must be calculated by process
                    with rank iRank
    """
    
    # Print error if less than one task per process
    if iTotal/iProc < 1: print('Error: Number of tasks smaller than number of processes')
    
    # If iTotal/iProc is not an integer, then all processes take on one additional task, except for the last process.
    iCeil = int(np.ceil(iTotal/iProc))
    lInt = []
    
    # Standard ordering from 0 to iTotal-1
    if bOrder:
        for i in range(iTotal):
            lInt.append(iRank * iCeil + i)
            if len(lInt) == iCeil or iRank * iCeil + i == iTotal-1:
                break
        
    # Using iProc steps
    else:
        for i in range(iTotal):
            lInt.append(iRank  + iProc * i)
            if iRank  + iProc * (i+1) > iTotal -1:
                break    
    
    return  np.array(lInt)
