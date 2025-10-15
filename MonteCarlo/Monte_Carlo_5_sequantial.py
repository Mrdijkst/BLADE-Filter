import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
# ...existing code...
import sys
import pathlib

# make the sibling folder "Volatility_model_progress" importable
_repo_root = pathlib.Path(__file__).resolve().parents[1]   # repo root
_vol_folder = _repo_root / "Volatility_model_progress"
if str(_vol_folder) not in sys.path:
    sys.path.insert(0, str(_vol_folder))

# import without the .py extension and using exact filename case
from Static_score_QLE9_best import RobustQLEVolatilityModel
from All_models import Beta_t_GARCH11
# ...existing code...


def run_combined_monte_carlo_simulation(num_repetitions=500, sample_size=3000):
    """
    Run Monte Carlo simulation for RobustQLEVolatilityModel comparing fixed alpha=0, fixed alpha=2,
    and estimated alpha on the same simulated datasets for direct comparison.
    
    Parameters:
    -----------
    num_repetitions : int
        Number of Monte Carlo repetitions
    sample_size : int
        Size of each simulated time series
        
    Returns:
    --------
    tuple of pandas.DataFrame
        Results of all Monte Carlo repetitions for all three approaches
    """
    # Set true parameters for data generation
    true_params = {
        'omega': 0.07,
        'gamma': 0.11,
        'beta': 0.80,
        'alpha_loss': 1.0,
        'c': 1.2
    }
    
    # Initialize arrays to store results
    fixed_alpha0_results = []
    fixed_alpha2_results = []
    estimated_results = []
    
    # Define column names for all result types
    # FIX: Add 'alpha_loss' column to fixed_columns
    fixed_columns = ['rep_id', 'omega', 'gamma', 'beta', 'alpha_loss', 'c', 'convergence', 'rmse', 'runtime', 'coverage']
    estimated_columns = ['rep_id', 'omega', 'gamma', 'beta', 'alpha_loss', 'c', 'convergence', 'rmse', 'runtime', 'coverage']
    
    # Start simulation
    print(f"Starting Combined Monte Carlo simulation with {num_repetitions} repetitions of {sample_size} observations each")
    start_time_total = time.time()
    
    for i in range(num_repetitions):
        rep_seed = 42 + i  # Different seed for each repetition
        
        # Create simulation model with true parameters
        sim_model = RobustQLEVolatilityModel(alpha_loss=true_params['alpha_loss'])
        sim_model.params = true_params.copy()
        
        # Simulate data - to be used by all model types
        y, true_vol = sim_model.simulate(sample_size, dist='n', seed=rep_seed)
        
        # ---------- Fixed Alpha=0 Model (Cauchy loss) ----------
        # Create model with fixed alpha=0
        fixed_alpha0_model = RobustQLEVolatilityModel(alpha_loss=0)
        
        # Time how long the estimation takes
        start_time_fixed0 = time.time()
        
        # Fit the fixed alpha=0 model
        try:
            fixed_alpha0_model.fit(y, method='Nelder-Mead', maxiter=1000, verbose=False)
            fixed_alpha0_convergence = True
        except Exception:
            fixed_alpha0_convergence = False
            
        fixed_alpha0_runtime = time.time() - start_time_fixed0
        
        # Calculate RMSE and coverage if convergence was successful
        if fixed_alpha0_convergence and fixed_alpha0_model.fitted_volatility is not None:
            fixed_alpha0_rmse = np.sqrt(np.mean((fixed_alpha0_model.fitted_volatility - true_vol)**2))
            
            # Calculate coverage probability correctly
            se_vol = np.sqrt(fixed_alpha0_model.fitted_volatility) / np.sqrt(sample_size)
            lower_bound = fixed_alpha0_model.fitted_volatility - 1.96 * se_vol
            upper_bound = fixed_alpha0_model.fitted_volatility + 1.96 * se_vol
            
            # Check if true volatility is within confidence interval
            coverage_check = np.logical_and(lower_bound <= true_vol, true_vol <= upper_bound)
            fixed_alpha0_coverage = np.mean(coverage_check)
        else:
            fixed_alpha0_rmse = np.nan
            fixed_alpha0_coverage = np.nan
        
        # Record results for fixed alpha=0 model
        fixed_alpha0_result = [i+1]  # rep_id
        
        if fixed_alpha0_convergence:
            for param in fixed_columns[1:-4]:  # Skip rep_id, convergence, rmse, runtime, coverage
                if param in fixed_alpha0_model.params:
                    fixed_alpha0_result.append(fixed_alpha0_model.params[param])
                else:
                    # For alpha_loss, append the fixed value 0
                    if param == 'alpha_loss':
                        fixed_alpha0_result.append(0)
                    else:
                        fixed_alpha0_result.append(np.nan)
        else:
            # If convergence failed, fill with NaNs
            fixed_alpha0_result.extend([np.nan] * (len(fixed_columns) - 4))
            
        fixed_alpha0_result.extend([fixed_alpha0_convergence, fixed_alpha0_rmse, fixed_alpha0_runtime, fixed_alpha0_coverage])
        fixed_alpha0_results.append(fixed_alpha0_result)
        
        # ---------- Fixed Alpha=2 Model (L₂ loss) ----------
        # Create model with fixed alpha=2
        fixed_alpha2_model = RobustQLEVolatilityModel(alpha_loss=2)
        
        # Time how long the estimation takes
        start_time_fixed2 = time.time()
        
        # Fit the fixed alpha=2 model
        try:
            fixed_alpha2_model.fit(y, method='Nelder-Mead', maxiter=1000, verbose=False)
            fixed_alpha2_convergence = True
        except Exception:
            fixed_alpha2_convergence = False
            
        fixed_alpha2_runtime = time.time() - start_time_fixed2
        
        # Calculate RMSE and coverage if convergence was successful
        if fixed_alpha2_convergence and fixed_alpha2_model.fitted_volatility is not None:
            fixed_alpha2_rmse = np.sqrt(np.mean((fixed_alpha2_model.fitted_volatility - true_vol)**2))
            
            # Calculate coverage probability correctly
            se_vol = np.sqrt(fixed_alpha2_model.fitted_volatility) / np.sqrt(sample_size)
            lower_bound = fixed_alpha2_model.fitted_volatility - 1.96 * se_vol
            upper_bound = fixed_alpha2_model.fitted_volatility + 1.96 * se_vol
            
            # Check if true volatility is within confidence interval
            coverage_check = np.logical_and(lower_bound <= true_vol, true_vol <= upper_bound)
            fixed_alpha2_coverage = np.mean(coverage_check)
        else:
            fixed_alpha2_rmse = np.nan
            fixed_alpha2_coverage = np.nan
        
        # Record results for fixed alpha=2 model
        fixed_alpha2_result = [i+1]  # rep_id
        
        if fixed_alpha2_convergence:
            for param in fixed_columns[1:-4]:  # Skip rep_id, convergence, rmse, runtime, coverage
                if param in fixed_alpha2_model.params:
                    fixed_alpha2_result.append(fixed_alpha2_model.params[param])
                else:
                    # For alpha_loss, append the fixed value 2
                    if param == 'alpha_loss':
                        fixed_alpha2_result.append(2)
                    else:
                        fixed_alpha2_result.append(np.nan)
        else:
            # If convergence failed, fill with NaNs
            fixed_alpha2_result.extend([np.nan] * (len(fixed_columns) - 4))
            
        fixed_alpha2_result.extend([fixed_alpha2_convergence, fixed_alpha2_rmse, fixed_alpha2_runtime, fixed_alpha2_coverage])
        fixed_alpha2_results.append(fixed_alpha2_result)
        
        # ---------- Estimated Alpha Model (Barron loss) ----------
        # Create model with estimated alpha
        estimated_model = RobustQLEVolatilityModel()
        
        # Time how long the estimation takes
        start_time_est = time.time()
        
        # Fit the estimated alpha model
        try:
            estimated_model.fit(y, method='Nelder-Mead', maxiter=1000, verbose=False)
            est_convergence = True
        except Exception:
            est_convergence = False
            
        est_runtime = time.time() - start_time_est
        
        # Calculate RMSE and coverage if convergence was successful
        if est_convergence and estimated_model.fitted_volatility is not None:
            est_rmse = np.sqrt(np.mean((estimated_model.fitted_volatility - true_vol)**2))
            
            # Calculate coverage probability correctly
            se_vol = np.sqrt(estimated_model.fitted_volatility) / np.sqrt(sample_size)
            lower_bound = estimated_model.fitted_volatility - 1.96 * se_vol
            upper_bound = estimated_model.fitted_volatility + 1.96 * se_vol
            
            # Check if true volatility is within confidence interval
            coverage_check = np.logical_and(lower_bound <= true_vol, true_vol <= upper_bound)
            est_coverage = np.mean(coverage_check)
        else:
            est_rmse = np.nan
            est_coverage = np.nan
        
        # Record results for estimated alpha model
        est_result = [i+1]  # rep_id
        
        if est_convergence:
            for param in estimated_columns[1:-4]:  # Skip rep_id, convergence, rmse, runtime, coverage
                if param in estimated_model.params:
                    est_result.append(estimated_model.params[param])
                else:
                    est_result.append(np.nan)
        else:
            # If convergence failed, fill with NaNs
            est_result.extend([np.nan] * (len(estimated_columns) - 4))
            
        est_result.extend([est_convergence, est_rmse, est_runtime, est_coverage])
        estimated_results.append(est_result)
        
        # Print progress
        if (i+1) % 50 == 0 or i+1 == num_repetitions:
            elapsed = time.time() - start_time_total
            print(f"Completed {i+1}/{num_repetitions} repetitions. Time elapsed: {elapsed:.2f} seconds")
    
    # Convert results to DataFrames
    fixed_alpha0_df = pd.DataFrame(fixed_alpha0_results, columns=fixed_columns)
    fixed_alpha2_df = pd.DataFrame(fixed_alpha2_results, columns=fixed_columns)
    estimated_df = pd.DataFrame(estimated_results, columns=estimated_columns)
    
    # Calculate and print summary statistics
    print_combined_summary(fixed_alpha0_df, fixed_alpha2_df, estimated_df, true_params)
    
    # Combined analysis - direct comparison between methods
    perform_paired_comparison(fixed_alpha0_df, fixed_alpha2_df, estimated_df)
    
    return fixed_alpha0_df, fixed_alpha2_df, estimated_df

def print_combined_summary(fixed_alpha0_df, fixed_alpha2_df, estimated_df, true_params):
    """Print summary statistics for fixed alpha=0, fixed alpha=2, and estimated alpha models"""
    
    # Print summary for fixed alpha=0 model (Cauchy loss)
    print("\n=== Fixed Alpha=0 Model (Cauchy loss) Results ===")
    print(f"Successful convergence rate: {fixed_alpha0_df['convergence'].mean()*100:.2f}%")
    print(f"Average runtime per repetition: {fixed_alpha0_df['runtime'].mean():.4f} seconds")
    print(f"Average RMSE of volatility: {fixed_alpha0_df['rmse'].mean():.6f}")
    print(f"Average coverage probability: {fixed_alpha0_df['coverage'].mean()*100:.2f}%")
    
    print("\nParameter Estimates (Mean ± Std):")
    for param in ['omega', 'gamma', 'beta', 'c']:
        if param in true_params:
            true_val = true_params[param]
            mean_val = fixed_alpha0_df[param].mean()
            std_val = fixed_alpha0_df[param].std()
            bias_val = mean_val - true_val
            print(f"{param}: {mean_val:.4f} ± {std_val:.4f} (True: {true_val:.4f}, Bias: {bias_val:.4f})")
    
    # Print summary for fixed alpha=2 model (L₂ loss)
    print("\n=== Fixed Alpha=2 Model (L₂ loss) Results ===")
    print(f"Successful convergence rate: {fixed_alpha2_df['convergence'].mean()*100:.2f}%")
    print(f"Average runtime per repetition: {fixed_alpha2_df['runtime'].mean():.4f} seconds")
    print(f"Average RMSE of volatility: {fixed_alpha2_df['rmse'].mean():.6f}")
    print(f"Average coverage probability: {fixed_alpha2_df['coverage'].mean()*100:.2f}%")
    
    print("\nParameter Estimates (Mean ± Std):")
    for param in ['omega', 'gamma', 'beta', 'c']:
        if param in true_params:
            true_val = true_params[param]
            mean_val = fixed_alpha2_df[param].mean()
            std_val = fixed_alpha2_df[param].std()
            bias_val = mean_val - true_val
            print(f"{param}: {mean_val:.4f} ± {std_val:.4f} (True: {true_val:.4f}, Bias: {bias_val:.4f})")
    
    # Print summary for estimated alpha model (Barron loss)
    print("\n=== Estimated Alpha Model (Barron loss) Results ===")
    print(f"Successful convergence rate: {estimated_df['convergence'].mean()*100:.2f}%")
    print(f"Average runtime per repetition: {estimated_df['runtime'].mean():.4f} seconds")
    print(f"Average RMSE of volatility: {estimated_df['rmse'].mean():.6f}")
    print(f"Average coverage probability: {estimated_df['coverage'].mean()*100:.2f}%")
    
    print("\nParameter Estimates (Mean ± Std):")
    for param in ['omega', 'gamma', 'beta', 'alpha_loss', 'c']:
        if param in true_params:
            true_val = true_params[param]
            mean_val = estimated_df[param].mean()
            std_val = estimated_df[param].std()
            bias_val = mean_val - true_val
            print(f"{param}: {mean_val:.4f} ± {std_val:.4f} (True: {true_val:.4f}, Bias: {bias_val:.4f})")

def perform_paired_comparison(fixed_alpha0_df, fixed_alpha2_df, estimated_df):
    """Perform a paired comparison between fixed alpha=0, fixed alpha=2, and estimated alpha models"""
    # Only include repetitions where all three models converged
    common_reps = np.intersect1d(
        np.intersect1d(
            fixed_alpha0_df.loc[fixed_alpha0_df['convergence'], 'rep_id'],
            fixed_alpha2_df.loc[fixed_alpha2_df['convergence'], 'rep_id']
        ),
        estimated_df.loc[estimated_df['convergence'], 'rep_id']
    )
    
    # Filter dataframes to only include common repetitions
    fixed_alpha0_common = fixed_alpha0_df[fixed_alpha0_df['rep_id'].isin(common_reps)]
    fixed_alpha2_common = fixed_alpha2_df[fixed_alpha2_df['rep_id'].isin(common_reps)]
    estimated_common = estimated_df[estimated_df['rep_id'].isin(common_reps)]
    
    # Sort by rep_id to ensure proper alignment
    fixed_alpha0_common = fixed_alpha0_common.sort_values('rep_id').reset_index(drop=True)
    fixed_alpha2_common = fixed_alpha2_common.sort_values('rep_id').reset_index(drop=True)
    estimated_common = estimated_common.sort_values('rep_id').reset_index(drop=True)
    
    # Calculate differences in key metrics
    rmse_diff_alpha0_est = fixed_alpha0_common['rmse'] - estimated_common['rmse']
    rmse_diff_alpha2_est = fixed_alpha2_common['rmse'] - estimated_common['rmse']
    rmse_diff_alpha0_alpha2 = fixed_alpha0_common['rmse'] - fixed_alpha2_common['rmse']
    
    runtime_diff_alpha0_est = fixed_alpha0_common['runtime'] - estimated_common['runtime']
    runtime_diff_alpha2_est = fixed_alpha2_common['runtime'] - estimated_common['runtime']
    runtime_diff_alpha0_alpha2 = fixed_alpha0_common['runtime'] - fixed_alpha2_common['runtime']
    
    coverage_diff_alpha0_est = fixed_alpha0_common['coverage'] - estimated_common['coverage']
    coverage_diff_alpha2_est = fixed_alpha2_common['coverage'] - estimated_common['coverage']
    coverage_diff_alpha0_alpha2 = fixed_alpha0_common['coverage'] - fixed_alpha2_common['coverage']
    
    # Create comparison dataframe for shared parameters
    comparison = pd.DataFrame({
        'rep_id': fixed_alpha0_common['rep_id'],
        'alpha0_rmse': fixed_alpha0_common['rmse'],
        'alpha2_rmse': fixed_alpha2_common['rmse'],
        'estimated_rmse': estimated_common['rmse'],
        'rmse_diff_alpha0_est': rmse_diff_alpha0_est,
        'rmse_diff_alpha2_est': rmse_diff_alpha2_est,
        'rmse_diff_alpha0_alpha2': rmse_diff_alpha0_alpha2,
        'alpha0_coverage': fixed_alpha0_common['coverage'],
        'alpha2_coverage': fixed_alpha2_common['coverage'],
        'estimated_coverage': estimated_common['coverage'],
        'coverage_diff_alpha0_est': coverage_diff_alpha0_est,
        'coverage_diff_alpha2_est': coverage_diff_alpha2_est,
        'coverage_diff_alpha0_alpha2': coverage_diff_alpha0_alpha2,
        'alpha0_runtime': fixed_alpha0_common['runtime'],
        'alpha2_runtime': fixed_alpha2_common['runtime'],
        'estimated_runtime': estimated_common['runtime'],
        'runtime_diff_alpha0_est': runtime_diff_alpha0_est,
        'runtime_diff_alpha2_est': runtime_diff_alpha2_est,
        'runtime_diff_alpha0_alpha2': runtime_diff_alpha0_alpha2
    })
    
    # Add parameter differences
    for param in ['omega', 'gamma', 'beta', 'c']:
        comparison[f'alpha0_{param}'] = fixed_alpha0_common[param]
        comparison[f'alpha2_{param}'] = fixed_alpha2_common[param]
        comparison[f'estimated_{param}'] = estimated_common[param]
        comparison[f'{param}_diff_alpha0_est'] = fixed_alpha0_common[param] - estimated_common[param]
        comparison[f'{param}_diff_alpha2_est'] = fixed_alpha2_common[param] - estimated_common[param]
        comparison[f'{param}_diff_alpha0_alpha2'] = fixed_alpha0_common[param] - fixed_alpha2_common[param]
    
    # Add estimated alpha_loss parameter
    comparison['estimated_alpha_loss'] = estimated_common['alpha_loss']
    
    # Print summary of direct comparison
    print("\n=== Direct Comparison ===")
    print(f"Number of common convergent repetitions: {len(common_reps)}")
    
    print("\n--- Fixed Alpha=0 (Cauchy) vs Estimated Alpha (Barron) ---")
    print(f"RMSE difference: {rmse_diff_alpha0_est.mean():.6f} ± {rmse_diff_alpha0_est.std():.6f}")
    print(f"Coverage probability difference: {coverage_diff_alpha0_est.mean()*100:.2f}% ± {coverage_diff_alpha0_est.std()*100:.2f}%")
    print(f"Runtime difference: {runtime_diff_alpha0_est.mean():.4f} ± {runtime_diff_alpha0_est.std():.4f} seconds")
    
    print("\n--- Fixed Alpha=2 (L₂) vs Estimated Alpha (Barron) ---")
    print(f"RMSE difference: {rmse_diff_alpha2_est.mean():.6f} ± {rmse_diff_alpha2_est.std():.6f}")
    print(f"Coverage probability difference: {coverage_diff_alpha2_est.mean()*100:.2f}% ± {coverage_diff_alpha2_est.std()*100:.2f}%")
    print(f"Runtime difference: {runtime_diff_alpha2_est.mean():.4f} ± {runtime_diff_alpha2_est.std():.4f} seconds")
    
    print("\n--- Fixed Alpha=0 (Cauchy) vs Fixed Alpha=2 (L₂) ---")
    print(f"RMSE difference: {rmse_diff_alpha0_alpha2.mean():.6f} ± {rmse_diff_alpha0_alpha2.std():.6f}")
    print(f"Coverage probability difference: {coverage_diff_alpha0_alpha2.mean()*100:.2f}% ± {coverage_diff_alpha0_alpha2.std()*100:.2f}%")
    print(f"Runtime difference: {runtime_diff_alpha0_alpha2.mean():.4f} ± {runtime_diff_alpha0_alpha2.std():.4f} seconds")
    
    # Compare parameter estimates
    print("\nParameter Estimate Differences (Fixed Alpha=0 - Estimated):")
    for param in ['omega', 'gamma', 'beta', 'c']:
        diff = comparison[f'{param}_diff_alpha0_est']
        print(f"{param}: {diff.mean():.4f} ± {diff.std():.4f}")
    
    print("\nParameter Estimate Differences (Fixed Alpha=2 - Estimated):")
    for param in ['omega', 'gamma', 'beta', 'c']:
        diff = comparison[f'{param}_diff_alpha2_est']
        print(f"{param}: {diff.mean():.4f} ± {diff.std():.4f}")
    
    print("\nParameter Estimate Differences (Fixed Alpha=0 - Fixed Alpha=2):")
    for param in ['omega', 'gamma', 'beta', 'c']:
        diff = comparison[f'{param}_diff_alpha0_alpha2']
        print(f"{param}: {diff.mean():.4f} ± {diff.std():.4f}")
    
    # Count how many times each model had better RMSE
    # Alpha=0 vs Estimated
    alpha0_better_than_est = sum(rmse_diff_alpha0_est < 0)
    est_better_than_alpha0 = sum(rmse_diff_alpha0_est > 0)
    alpha0_est_tied = sum(rmse_diff_alpha0_est == 0)
    
    # Alpha=2 vs Estimated
    alpha2_better_than_est = sum(rmse_diff_alpha2_est < 0)
    est_better_than_alpha2 = sum(rmse_diff_alpha2_est > 0)
    alpha2_est_tied = sum(rmse_diff_alpha2_est == 0)
    
    # Alpha=0 vs Alpha=2
    alpha0_better_than_alpha2 = sum(rmse_diff_alpha0_alpha2 < 0)
    alpha2_better_than_alpha0 = sum(rmse_diff_alpha0_alpha2 > 0)
    alpha0_alpha2_tied = sum(rmse_diff_alpha0_alpha2 == 0)
    
    print(f"\nRMSE comparison (Alpha=0 vs Estimated): Alpha=0 better in {alpha0_better_than_est} cases ({alpha0_better_than_est/len(common_reps)*100:.2f}%), Estimated better in {est_better_than_alpha0} cases ({est_better_than_alpha0/len(common_reps)*100:.2f}%), Tied in {alpha0_est_tied} cases ({alpha0_est_tied/len(common_reps)*100:.2f}%)")
    print(f"RMSE comparison (Alpha=2 vs Estimated): Alpha=2 better in {alpha2_better_than_est} cases ({alpha2_better_than_est/len(common_reps)*100:.2f}%), Estimated better in {est_better_than_alpha2} cases ({est_better_than_alpha2/len(common_reps)*100:.2f}%), Tied in {alpha2_est_tied} cases ({alpha2_est_tied/len(common_reps)*100:.2f}%)")
    print(f"RMSE comparison (Alpha=0 vs Alpha=2): Alpha=0 better in {alpha0_better_than_alpha2} cases ({alpha0_better_than_alpha2/len(common_reps)*100:.2f}%), Alpha=2 better in {alpha2_better_than_alpha0} cases ({alpha2_better_than_alpha0/len(common_reps)*100:.2f}%), Tied in {alpha0_alpha2_tied} cases ({alpha0_alpha2_tied/len(common_reps)*100:.2f}%)")
    
    # Find best model overall by RMSE
    best_model_counts = {
        'Alpha=0 (Cauchy)': sum((fixed_alpha0_common['rmse'] < fixed_alpha2_common['rmse']) & 
                               (fixed_alpha0_common['rmse'] < estimated_common['rmse'])),
        'Alpha=2 (L₂)': sum((fixed_alpha2_common['rmse'] < fixed_alpha0_common['rmse']) & 
                           (fixed_alpha2_common['rmse'] < estimated_common['rmse'])),
        'Estimated (Barron)': sum((estimated_common['rmse'] < fixed_alpha0_common['rmse']) & 
                                 (estimated_common['rmse'] < fixed_alpha2_common['rmse']))
    }
    
    print("\n=== Overall Best Model by RMSE ===")
    for model, count in best_model_counts.items():
        print(f"{model}: Best in {count} cases ({count/len(common_reps)*100:.2f}%)")
    
    # Calculate and print distribution of estimated alpha values
    print("\n=== Distribution of Estimated Alpha Values ===")
    alpha_mean = estimated_common['alpha_loss'].mean()
    alpha_std = estimated_common['alpha_loss'].std()
    alpha_median = estimated_common['alpha_loss'].median()
    alpha_min = estimated_common['alpha_loss'].min()
    alpha_max = estimated_common['alpha_loss'].max()
    
    print(f"Mean: {alpha_mean:.4f}")
    print(f"Std Dev: {alpha_std:.4f}")
    print(f"Median: {alpha_median:.4f}")
    print(f"Min: {alpha_min:.4f}")
    print(f"Max: {alpha_max:.4f}")
    
    # Calculate percentiles of estimated alpha
    percentiles = [10, 25, 50, 75, 90]
    alpha_percentiles = np.percentile(estimated_common['alpha_loss'], percentiles)
    print("\nPercentiles of estimated alpha:")
    for p, val in zip(percentiles, alpha_percentiles):
        print(f"{p}%: {val:.4f}")
    
    return comparison

if __name__ == "__main__":
    # Run the simulation
    fixed_alpha0_results, fixed_alpha2_results, estimated_results = run_combined_monte_carlo_simulation(
        num_repetitions=10, 
        sample_size=3000
    )
    
    # Create LaTeX tables
    true_params = {
        'omega': 0.07,
        'gamma': 0.11,
        'beta': 0.80,
        'alpha_loss': 1.0,
        'c': 1.2
    }
    
    # Save results to CSV files for further analysis
    fixed_alpha0_results.to_csv('fixed_alpha0_results.csv', index=False)
    fixed_alpha2_results.to_csv('fixed_alpha2_results.csv', index=False)
    estimated_results.to_csv('estimated_results.csv', index=False)
    
    print("Simulation completed and results saved to files.")