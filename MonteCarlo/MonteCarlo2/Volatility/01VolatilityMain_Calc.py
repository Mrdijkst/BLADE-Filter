
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Mainfile Location Experiments MC
"""

###########################################################
### Imports

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables

# MPI
from mpi4py import MPI

# System
#import os

# Plots
#import matplotlib.pyplot as plt
#from matplotlib import rc
#os.environ["PATH"] += os.pathsep + '/usr/local/bin' 
#os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':'15'})
#rc('text', usetex=False)
#%matplotlib qt

# Dependencies
from Volatility.Volatility_Ramon import *     
  

###########################################################
def ProcWrapper(vInt, dNu, dEps, sExperiment, vGamma1Grid, iTTrain, iTTest):
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
    
      
    lGamma2 = [1, 1.5, 2, 4, "Loglik"]
    iNumBenchmarks = 1
    mLoss = np.zeros((len(lGamma2), len(vGamma1Grid) + iNumBenchmarks, len(vInt)))
    
    for i in range(len(vInt)):
        iSeed = int(123 + int(vInt[i]))
        
        # No distortion
        if sExperiment == 'eps1':
            mLoss[:,:,i] =  run_one_simulation_scale_train_contaminated(
                dEps,
                vGamma1Grid,
                iTTrain,
                iTTest,
                iSeed,
            )
            
        if sExperiment == 'eps0.99':
            mLoss[:,:,i] =  run_one_simulation_scale_train_contaminated(
                dNu,
                dEps,
                vGamma1Grid,
                iTTrain,
                iTTest, 
                iSeed,
            )
            
        # Distortion
        if sExperiment == 'eps0.98':
            mLoss[:,:,i] =  run_one_simulation_scale_train_contaminated(
                dEps,
                vGamma1Grid,
                iTTrain,
                iTTest,
                iSeed,
            )
            
        if sExperiment == 'eps0.97':
            mLoss[:,:,i] =  run_one_simulation_scale_train_contaminated(
                dEps,
                vGamma1Grid,
                iTTrain,
                iTTest,
                iSeed,
            )

        elif sExperiment == 'nu5_Distortion':    
            print('Experiment not supported')        
    
    return mLoss


###########################################################  
### main
def main():    
    
    ########################################
    ### Parameters
    ########################################
   
    ## Fundamentals
    
    # Experiment
    #sExperiment = 'eps1'
    sExperiment = 'eps0.99'
    #sExperiment = 'eps0.98'
    #sExperiment = 'eps0.97'
    
    # Main parameter
    
    dEps = sExperiment.split('_')[0][3:] # extract eps from experiment name
    dNu = 10
    
    # Other parameters
    iRep = 1000                  # number of Monte Carlo replications
    iTTrain=2500
    iTTest=1000  
    iLenGamma1Grid = 400
   
    # Initialization
    bOrderMPI = False           # order task distribution [sequentially if true]
    vGamma1Grid = np.linspace(-10.0, 2.0, iLenGamma1Grid)
    sExperimentName = sExperiment + '_dNu' + str(dNu) + '_Eps' + str(dEps)
    
  
    ## Initialisation MPI device
    comm = MPI.COMM_WORLD
    iRank = comm.Get_rank()  # rank process
    iProc = comm.Get_size()  # total number of processes 
    
    lGamma2 = [1, 1.5, 2, 4, "Loglik"]
    iNumBenchmarks = 1
    
    
    # Distribute
    vInt = MPITaskDistributor(iRank, iProc, iRep, bOrder=bOrderMPI) # calculate indices simulations assigned to this process
   

    comm.barrier() # wait for everyone
    if iRank ==0: print('>> MPI program activated')
      
    # DM calculations
    mLoss = np.zeros((len(lGamma2), len(vGamma1Grid) + iNumBenchmarks, iRep)) 
    mLossProc = ProcWrapper(vInt, dNu, dEps, sExperiment, vGamma1Grid, iTTrain, iTTest)
    lLoss = comm.gather(mLossProc, root=0)
    comm.barrier() # wait for everyone
        
    ## Combine data on root 
    if iRank == 0:
        iCount = 0
        for rank in range(len(lLoss)):
            vIntRank = MPITaskDistributor(rank, iProc, iRep, bOrder=bOrderMPI) 
            for i in range(vIntRank.size): 
                mLoss[:,:,vIntRank[i]] = lLoss[rank][:,:,i]
                    
        print(mLoss[:,:,0])

        np.save('mLoss/mLoss_' + sExperimentName + '.npy' , mLoss)
    
   
###########################################################
### start main
if __name__ == "__main__":
    main()
