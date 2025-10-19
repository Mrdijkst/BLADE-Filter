import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.special import gammaln
import matplotlib.pyplot as plt

from All_models import RobustQLEModel

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
        init_params = np.array([0.07, 0.1, 0.8,6.0])  # [ω, α, β, ν]
        
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
    

def run_single_simulation(rep_id, sample_size=3000, dgp_df=10):
    """
    Run one Monte Carlo replication with outliers and fit Beta-t-GARCH(1,1).
    """
    true_params = {
        'omega': 0.07,
        'gamma': 0.11,
        'beta': 0.80,
        'alpha_loss': 1.0,
        'c': 1.2
    }

    rep_seed = 42 + rep_id +500

    # simulate clean data
    sim_model = RobustQLEModel(model_type='volatility', alpha_loss=true_params['alpha_loss'])
    sim_model.params = true_params
    y_clean, true_vol = sim_model.simulate(sample_size, dist='n', df=dgp_df, seed=rep_seed)

    # inject outliers
    y_outliers = y_clean.copy()
    np.random.seed(rep_seed)
    outlier_positions = np.random.choice(range(500, sample_size-500), 25, replace=False)
    outlier_signs = np.random.choice([-1, 1], len(outlier_positions))

    for i, pos in enumerate(outlier_positions):
        local_std = np.sqrt(true_vol[pos])
        y_outliers[pos] = outlier_signs[i] * np.random.uniform(6, 10) * local_std

    y = y_outliers

    betat_columns = ['rep_id', 'omega', 'alpha', 'beta', 'nu',
                     'convergence', 'rmse', 'mae', 'runtime']

    res_betat = [rep_id+1]
    start = time.time()
    try:
        bt = Beta_t_GARCH11(y)
        result = bt.fit()
        fhat = bt.get_fitted_variances()
        conv = result.success
        rmse = float(np.sqrt(np.mean((fhat - true_vol)**2)))
        mae  = float(np.mean(np.abs(fhat - true_vol)))
        omega, alpha_like, beta, nu = bt.params
    except Exception:
        conv, rmse, mae = False, np.nan, np.nan
        omega = alpha_like = beta = nu = np.nan
    runtime = time.time() - start

    res_betat.extend([omega, alpha_like, beta, nu])
    res_betat.extend([conv, rmse, mae, runtime])

    return {'betat': res_betat, 'columns': {'betat': betat_columns}}


def run_parallel_monte_carlo_simulation(num_repetitions, sample_size, num_processes=None, dgp_df=5):
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"Starting parallel Monte Carlo with {num_repetitions} reps, T={sample_size}, procs={num_processes}")
    start_time_total = time.time()

    pool = mp.Pool(processes=num_processes)
    worker_func = partial(run_single_simulation, sample_size=sample_size, dgp_df=dgp_df)

    results = list(tqdm(pool.imap(worker_func, range(num_repetitions)),
                        total=num_repetitions,
                        desc="Monte Carlo Progress"))

    pool.close()
    pool.join()

    betat_rows = [r['betat'] for r in results]
    cols_betat = results[0]['columns']['betat']
    df_betat   = pd.DataFrame(betat_rows, columns=cols_betat)

    elapsed = time.time() - start_time_total
    print(f"Completed {num_repetitions} repetitions in {elapsed:.2f} seconds")

    # ---- compute average RMSE ----
    avg_rmse = df_betat['rmse'].mean(skipna=True)
    print(f"Average RMSE over {num_repetitions} reps: {avg_rmse:.6f}")

    # Optionally append to the dataframe as a one-row summary
    summary_row = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan,
                                 np.nan, avg_rmse, np.nan, np.nan]],
                               columns=cols_betat)
    df_betat = pd.concat([df_betat, summary_row], ignore_index=True)

    return df_betat, avg_rmse


if __name__ == "__main__":
    num_repetitions = 5
    sample_size = 4000
    num_cores = mp.cpu_count()
    print(f"Available CPU cores: {num_cores}")
    num_processes = max(1, int(num_cores * 0.8))

    df_betat, avg_rmse = run_parallel_monte_carlo_simulation(
        num_repetitions=num_repetitions,
        sample_size=sample_size,
        num_processes=num_processes,
        dgp_df=5
    )

    df_betat.to_csv('outlier_monte_carlo_betatgarch_results.csv', index=False)
    print(f"Results saved. Average RMSE: {avg_rmse:.6f}")
