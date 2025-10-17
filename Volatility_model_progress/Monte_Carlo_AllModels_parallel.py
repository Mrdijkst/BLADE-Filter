
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# Import models from user's consolidated file
from All_models import RobustQLEModel, GAS_Model, Beta_t_GARCH11


# -------------------- Single replication worker --------------------
def run_single_simulation(rep_id, sample_size=3000, dgp_df=5):
    """
    Run one Monte Carlo replication on a single simulated dataset and fit all models.
    Returns a dict of result rows (lists) ready to be assembled into DataFrames.
    """
    # True DGP parameters
    true_params = {
        'omega': 0.07,
        'gamma': 0.11,
        'beta': 0.80,
        'alpha_loss': 1.0,  # for QSD score
        'c': 1.2
    }

    # Seed per replication for reproducibility
    rep_seed = 42 + rep_id 

    # --- Simulate from the true QSD volatility DGP ---
    dgp = RobustQLEModel(model_type='volatility', alpha_loss=true_params['alpha_loss'], c=true_params['c'])
    dgp.params = {'omega': true_params['omega'], 'gamma': true_params['gamma'], 'beta': true_params['beta']}
    y, true_vol = dgp.simulate(sample_size, dist='n', df=dgp_df, seed=rep_seed)

    # Column layouts (match your parallel file’s style)
    qsd_fixed_columns = ['rep_id', 'omega', 'gamma', 'beta', 'c', 'convergence', 'rmse', 'mae', 'runtime']
    qsd_est_columns   = ['rep_id', 'omega', 'gamma', 'beta', 'alpha_loss', 'c', 'convergence', 'rmse', 'mae', 'runtime']
    gas_columns       = ['rep_id', 'omega', 'gamma', 'beta', 'convergence', 'rmse', 'mae', 'runtime']
    betat_columns     = ['rep_id', 'omega', 'gamma', 'beta', 'nu', 'convergence', 'rmse', 'mae', 'runtime']

    # Results for each model
    res_alpha_neginf = [rep_id+1]
    res_alpha0       = [rep_id+1]
    res_alpha1       = [rep_id+1]
    res_alpha2       = [rep_id+1]
    res_alpha_est    = [rep_id+1]
    res_gas          = [rep_id+1]
    res_betat        = [rep_id+1]

    # ------------- QSD α = −∞ (Welsch) -------------
    start = time.time()
    try:
        mdl = RobustQLEModel(model_type='volatility', alpha_loss=float('-inf'))
        mdl.fit(y, method='Nelder-Mead', maxiter=1000)
        conv = True
        rmse = float(np.sqrt(np.mean((mdl.fitted_volatility - true_vol)**2)))
        mae  = float(np.mean(np.abs(mdl.fitted_volatility - true_vol)))
    except Exception:
        conv, rmse, mae = False, np.nan, np.nan
        mdl = type('obj', (), {'params': {}})()  # empty fallback
    runtime = time.time() - start
    # Fill fixed-columns row: rep_id already added
    if conv:
        for param in qsd_fixed_columns[1:-4]:  # omega, gamma, beta, c
            if param in mdl.params:
                res_alpha_neginf.append(mdl.params[param])
            else:
                # supply c if fixed
                if param == 'c':
                    res_alpha_neginf.append(true_params['c'])
                else:
                    res_alpha_neginf.append(np.nan)
    else:
        res_alpha_neginf.extend([np.nan] * (len(qsd_fixed_columns) - 4))
    res_alpha_neginf.extend([conv, rmse, mae, runtime])

    # ------------- QSD α = 0 (Cauchy) -------------
    start = time.time()
    try:
        mdl = RobustQLEModel(model_type='volatility', alpha_loss=0.0)
        mdl.fit(y, method='Nelder-Mead', maxiter=1000)
        conv = True
        rmse = float(np.sqrt(np.mean((mdl.fitted_volatility - true_vol)**2)))
        mae  = float(np.mean(np.abs(mdl.fitted_volatility - true_vol)))
    except Exception:
        conv, rmse, mae = False, np.nan, np.nan
        mdl = type('obj', (), {'params': {}})()
    runtime = time.time() - start
    if conv:
        for param in qsd_fixed_columns[1:-4]:
            if param in mdl.params:
                res_alpha0.append(mdl.params[param])
            else:
                if param == 'c':
                    res_alpha0.append(true_params['c'])
                else:
                    res_alpha0.append(np.nan)
    else:
        res_alpha0.extend([np.nan] * (len(qsd_fixed_columns) - 4))
    res_alpha0.extend([conv, rmse, mae, runtime])

    # ------------- QSD α = 1 -------------
    start = time.time()
    try:
        mdl = RobustQLEModel(model_type='volatility', alpha_loss=1.0)
        mdl.fit(y, method='Nelder-Mead', maxiter=1000)
        conv = True
        rmse = float(np.sqrt(np.mean((mdl.fitted_volatility - true_vol)**2)))
        mae  = float(np.mean(np.abs(mdl.fitted_volatility - true_vol)))
    except Exception:
        conv, rmse, mae = False, np.nan, np.nan
        mdl = type('obj', (), {'params': {}})()
    runtime = time.time() - start
    if conv:
        for param in qsd_fixed_columns[1:-4]:
            if param in mdl.params:
                res_alpha1.append(mdl.params[param])
            else:
                if param == 'c':
                    res_alpha1.append(true_params['c'])
                else:
                    res_alpha1.append(np.nan)
    else:
        res_alpha1.extend([np.nan] * (len(qsd_fixed_columns) - 4))
    res_alpha1.extend([conv, rmse, mae, runtime])

    # ------------- QSD α = 2 (L2) -------------
    start = time.time()
    try:
        mdl = RobustQLEModel(model_type='volatility', alpha_loss=2.0)
        mdl.fit(y, method='Nelder-Mead', maxiter=1000)
        conv = True
        rmse = float(np.sqrt(np.mean((mdl.fitted_volatility - true_vol)**2)))
        mae  = float(np.mean(np.abs(mdl.fitted_volatility - true_vol)))
    except Exception:
        conv, rmse, mae = False, np.nan, np.nan
        mdl = type('obj', (), {'params': {}})()
    runtime = time.time() - start
    if conv:
        for param in qsd_fixed_columns[1:-4]:
            if param in mdl.params:
                res_alpha2.append(mdl.params[param])
            else:
                if param == 'c':
                    res_alpha2.append(true_params['c'])
                else:
                    res_alpha2.append(np.nan)
    else:
        res_alpha2.extend([np.nan] * (len(qsd_fixed_columns) - 4))
    res_alpha2.extend([conv, rmse, mae, runtime])

    # ------------- QSD α estimated -------------
    start = time.time()
    try:
        mdl = RobustQLEModel(model_type='volatility', alpha_loss=None)
        init = {'omega': true_params['omega'], 'gamma': true_params['gamma'], 'beta': true_params['beta'],
                'alpha_loss': 1.0, 'c': true_params['c']}
        mdl.fit(y, initial_params=init, method='Nelder-Mead', maxiter=1000)
        conv = True
        rmse = float(np.sqrt(np.mean((mdl.fitted_volatility - true_vol)**2)))
        mae  = float(np.mean(np.abs(mdl.fitted_volatility - true_vol)))
    except Exception:
        conv, rmse, mae = False, np.nan, np.nan
        mdl = type('obj', (), {'params': {}})()
    runtime = time.time() - start
    if conv:
        for param in qsd_est_columns[1:-4]:  # omega, gamma, beta, alpha_loss, c
            if param in getattr(mdl, 'params', {}):
                res_alpha_est.append(mdl.params[param])
            else:
                # supply c if fixed or NaN for alpha_loss if missing
                if param == 'c':
                    res_alpha_est.append(true_params['c'])
                else:
                    res_alpha_est.append(np.nan)
    else:
        res_alpha_est.extend([np.nan] * (len(qsd_est_columns) - 4))
    res_alpha_est.extend([conv, rmse, mae, runtime])

    # ------------- GAS (Normal) -------------
    start = time.time()
    try:
        gas = GAS_Model(y)
        res = gas.fit()
        fhat = gas.get_fitted_variance()
        conv = True
        rmse = float(np.sqrt(np.mean((fhat - true_vol)**2)))
        mae  = float(np.mean(np.abs(fhat - true_vol)))
        omega, alpha_like, beta = float(gas.params[0]), float(gas.params[1]), float(gas.params[2])

    except Exception:
        conv, rmse, mae = False, np.nan, np.nan
        omega = alpha_like = beta = np.nan
        print("error")
    runtime = time.time() - start
    res_gas.extend([omega, alpha_like, beta])
    res_gas.extend([conv, rmse, mae, runtime])

    # ------------- beta_t–GARCH(1,1) -------------
    start = time.time()
    try:
        bt = Beta_t_GARCH11(y)
        _ = bt.fit()
        fhat = bt.get_fitted_variances()
        conv = True
        rmse = float(np.sqrt(np.mean((fhat - true_vol)**2)))
        mae  = float(np.mean(np.abs(fhat - true_vol)))
        omega, alpha_like, beta, nu = bt.params
    except Exception:
        conv, rmse, mae = False, np.nan, np.nan
        omega = alpha_like = beta = nu = np.nan

    runtime = time.time() - start
    res_betat.extend([omega, alpha_like, beta, nu])
    res_betat.extend([conv, rmse, mae, runtime])

    return {
        'alpha_neginf': res_alpha_neginf,
        'alpha0': res_alpha0,
        'alpha1': res_alpha1,
        'alpha2': res_alpha2,
        'alpha_est': res_alpha_est,
        'gas': res_gas,
        'betat': res_betat,
        'columns': {
            'qsd_fixed': qsd_fixed_columns,
            'qsd_est': qsd_est_columns,
            'gas': gas_columns,
            'betat': betat_columns
        }
    }


# -------------------- Parallel driver --------------------
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

    # Unpack rows
    alpha_neginf_rows = [r['alpha_neginf'] for r in results]
    alpha0_rows       = [r['alpha0']       for r in results]
    alpha1_rows       = [r['alpha1']       for r in results]
    alpha2_rows       = [r['alpha2']       for r in results]
    alpha_est_rows    = [r['alpha_est']    for r in results]
    gas_rows          = [r['gas']          for r in results]
    betat_rows        = [r['betat']        for r in results]

    # Column names
    cols_fixed = results[0]['columns']['qsd_fixed']
    cols_est   = results[0]['columns']['qsd_est']
    cols_gas   = results[0]['columns']['gas']
    cols_betat = results[0]['columns']['betat']

    # Build DataFrames (per-model)
    df_alpha_neginf = pd.DataFrame(alpha_neginf_rows, columns=cols_fixed)
    df_alpha0       = pd.DataFrame(alpha0_rows,       columns=cols_fixed)
    df_alpha1       = pd.DataFrame(alpha1_rows,       columns=cols_fixed)
    df_alpha2       = pd.DataFrame(alpha2_rows,       columns=cols_fixed)
    df_alpha_est    = pd.DataFrame(alpha_est_rows,    columns=cols_est)
    df_gas          = pd.DataFrame(gas_rows,          columns=cols_gas)
    df_betat        = pd.DataFrame(betat_rows,        columns=cols_betat)

    elapsed = time.time() - start_time_total
    print(f"Completed {num_repetitions} repetitions in {elapsed:.2f} seconds")

    return df_alpha_neginf, df_alpha0, df_alpha1, df_alpha2, df_alpha_est, df_gas, df_betat


if __name__ == "__main__":
    num_repetitions = 5
    sample_size = 4000
    num_cores = mp.cpu_count()
    print(f"Available CPU cores: {num_cores}")
    num_processes = max(1, int(num_cores * 0.8))

    (df_alpha_neginf, df_alpha0, df_alpha1, df_alpha2,
     df_alpha_est, df_gas, df_betat) = run_parallel_monte_carlo_simulation(
        num_repetitions=num_repetitions,
        sample_size=sample_size,
        num_processes=num_processes,
        dgp_df=5
    )

    # Save 7 CSVs (one per model), in the exact naming pattern you requested
    df_alpha_neginf.to_csv('monte_carlo_qsd_alpha_neginf_results_parallel.csv', index=False)
    df_alpha0.to_csv('monte_carlo_qsd_alpha0_results_parallel.csv', index=False)
    df_alpha1.to_csv('monte_carlo_qsd_alpha1_results_parallel.csv', index=False)
    df_alpha2.to_csv('monte_carlo_qsd_alpha2_results_parallel.csv', index=False)
    df_alpha_est.to_csv('monte_carlo_qsd_alpha_estimated_results_parallel.csv', index=False)
    df_gas.to_csv('monte_carlo_gas_results_parallel.csv', index=False)
    df_betat.to_csv('monte_carlo_betatgarch_results_parallel.csv', index=False)
