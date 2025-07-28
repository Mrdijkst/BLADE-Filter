import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
import time
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from Static_score_QLE9_best import RobustQLEVolatilityModel


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


    def _compute_nll_and_variances(self, params):
        """
        Given params = [omega, alpha, beta], construct {f_t}_{t=0..T-1} in‐sample 
        and return (negative log‐likelihood, the array f).
        """
        ω, α, β, ν = params

        
        # If any basic constraints are violated, return a large penalty
        if ω <= 0 or α < 0 or β < 0 or α + β >= 1:
            return 1e10, None
        
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
        We impose:
          • ω > 0,
          • α ≥ 0,
          • β ≥ 0,
          • α + β < 1   (weak stationarity).
        """
        # Initial guess: set ω ≈ 0.1 × Var(y), α = 0.05, β = 0.9
        sample_var = np.var(self.y)
        init_params = np.array([0.07, 0.11, 0.8,6.0])  # [ω, α, β, ν]
        
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
            method='nelder-mead',
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




def run_single_simulation(rep_id, sample_size=2000):
    """
    Function to run a single Monte Carlo simulation for β_t GARCH(1,1) model.
    This function is designed to be used with multiprocessing.
    """
    # Set true parameters for data generation
    true_params = {
        'omega': 0.07,
        'gamma': 0.11,
        'beta': 0.80,
        'alpha_loss': 1,
        'c': 1.2
    }
    
    rep_seed = 42 + rep_id   # Different seed for each repetition
    
    # Create simulation model with true parameters
    sim_model = RobustQLEVolatilityModel(alpha_loss=true_params['alpha_loss'])
    sim_model.params = true_params.copy()

    
    # Simulate clean data
    T = sample_size
    y, true_vol = sim_model.simulate(T, dist='n', df=7, seed=rep_seed)
    

    

    # Define column names
    columns = ['rep_id', 'omega', 'gamma', 'beta', 'nu', 'convergence', 'rmse', 'runtime']
    
    # Fit the model
    model = Beta_t_GARCH11(y)
    start_time = time.time()
    
    
    result = model.fit()
    fitted_params = result.x
    convergence = True
   
        
    runtime = time.time() - start_time
    
    # Calculate RMSE
    if convergence and model.get_fitted_variances() is not None:
        rmse = np.sqrt(np.mean((model.get_fitted_variances() - true_vol)**2))
    else:
        rmse = np.nan
    
    # Record results
    result = [rep_id + 1]  # rep_id
    
    if convergence and fitted_params is not None:
        result.extend(fitted_params)
    else:
        # If convergence failed, fill with NaNs
        result.extend([np.nan, np.nan, np.nan, np.nan])
        
    result.extend([convergence, rmse, runtime])
    
    return result


def run_parallel_monte_carlo_simulation(num_repetitions, sample_size, num_processes=None):
    """
    Run Monte Carlo simulation in parallel using multiple CPU cores for β_t GARCH(1,1) model
    
    Parameters:
    -----------
    num_repetitions : int
        Number of Monte Carlo repetitions
    sample_size : int
        Size of each simulated time series
    num_processes : int, optional
        Number of processes to use. Default is None (use all available cores)
        
    Returns:
    --------
    pandas.DataFrame
        Results of all Monte Carlo repetitions
    """
    # Set number of processes (use all available if not specified)
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Starting parallel Monte Carlo simulation with {num_repetitions} repetitions using {num_processes} processes")
    print("Running simulations for β_t GARCH(1,1) model")
    start_time_total = time.time()
    
    # Create process pool
    pool = mp.Pool(processes=num_processes)
    
    # Create function with fixed sample_size
    worker_func = partial(run_single_simulation, sample_size=sample_size)
    
    # Run simulations in parallel with progress bar
    results = list(tqdm(pool.imap(worker_func, range(num_repetitions)), 
                         total=num_repetitions, 
                         desc="Monte Carlo Progress"))
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Column names for DataFrame
    columns = ['rep_id', 'omega', 'gamma', 'beta', 'nu', 'convergence', 'rmse', 'runtime']
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=columns)
    
    # Calculate elapsed time
    elapsed = time.time() - start_time_total
    print(f"Completed {num_repetitions} repetitions in {elapsed:.2f} seconds")
    
    # Calculate speed improvement compared to sequential processing
    avg_time_per_sim = results_df['runtime'].mean()
    estimated_sequential_time = avg_time_per_sim * num_repetitions
    speedup = estimated_sequential_time / elapsed
    print(f"Estimated speedup factor: {speedup:.2f}x")
    
    # Print some summary statistics
    convergence_rate = results_df['convergence'].mean() * 100
    print(f"Convergence rate: {convergence_rate:.1f}%")
    
    if convergence_rate > 0:
        converged_results = results_df[results_df['convergence'] == True]
        print(f"Average RMSE (converged simulations): {converged_results['rmse'].mean():.6f}")
        print(f"Parameter estimates (mean ± std):")
        for param in ['omega', 'gamma', 'beta', 'nu']:
            mean_val = converged_results[param].mean()
            std_val = converged_results[param].std()
            print(f"  {param}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Print true parameter values for comparison
        true_params = {'omega': 0.07, 'gamma': 0.11, 'beta': 0.80, 'nu': 5.0}
        print(f"\nTrue parameter values:")
        for param in ['omega', 'gamma', 'beta', 'nu']:
            print(f"  {param}: {true_params[param]:.4f}")
    
    return results_df


if __name__ == "__main__":
    # Number of repetitions and sample size
    num_repetitions = 1000
    sample_size = 3000
    
    # Get number of available CPU cores
    num_cores = mp.cpu_count()
    print(f"Available CPU cores: {num_cores}")
    
    # Run parallel MC simulation (using 80% of available cores)
    num_processes = max(1, int(num_cores * 0.8))
    results_df = run_parallel_monte_carlo_simulation(
        num_repetitions=num_repetitions, 
        sample_size=sample_size,
        num_processes=num_processes
    )
    
    # Save results to CSV file
    output_filename = f'monte_carlo_beta_t_garch_results_parallel_{sample_size}_{num_repetitions}.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to: {output_filename}")
    
    # Display first few rows as verification
    print("\nFirst 5 rows of results:")
    print(results_df.head())
    
    # Additional analysis for converged results
    converged_results = results_df[results_df['convergence'] == True]
    if len(converged_results) > 0:
        print(f"\nDetailed statistics for {len(converged_results)} converged simulations:")
        print(converged_results[['omega', 'gamma', 'beta', 'nu', 'rmse']].describe())