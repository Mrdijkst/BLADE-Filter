import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from Static_score_QLE9_best import RobustQLEVolatilityModel

class RobustQLEVolatilityModel:
    def __init__(self, alpha_loss: float = None, c: float = None):
        self.alpha_loss = alpha_loss
        self.c = c
        self.params = None
        self.param_names = ['omega', 'gamma', 'beta']
        if alpha_loss is None:
            self.param_names.append('alpha_loss')
        if c is None:
            self.param_names.append('c')
        self.fitted_volatility = None
        self.residuals = None
    
    def _rho_derivative(self, e: np.ndarray, alpha: float, c: float) -> np.ndarray:
        # Handle scalar inputs by converting to numpy arrays
        if np.isscalar(e):
            e = np.array([e])
            scalar_input = True
        else:
            scalar_input = False
            
        if alpha == 2:
            # L2 loss (least squares)
            result = e / (c**2)
        elif alpha == 0:
            # Cauchy/Lorentzian loss
            result = (2 * e) / (e**2 + 2 * c**2)
        elif alpha == float('-inf'):
            # Welsch/Leclerc loss
            result = (e / c**2) * np.exp(-0.5 * (e/c)**2)
        else:
            # General case
            result = (e / c**2) * np.power((e**2 / c**2) / np.abs(alpha-2) + 1, alpha/2 - 1)
        
        # Return scalar if input was scalar
        if scalar_input:
            return result[0]
        return result
    
    def _rho_second_derivative(self, e: np.ndarray, alpha: float, c: float) -> np.ndarray:
        # Handle scalar inputs by converting to numpy arrays
        if np.isscalar(e):
            e = np.array([e])
            scalar_input = True
        else:
            scalar_input = False
            
        if alpha == 2:
            # L2 loss (least squares)
            result = -np.ones_like(e) / (c**2)
        elif alpha == 0:
            # Cauchy/Lorentzian loss
            denom = (e**2 + 2 * c**2)**2
            result = -(2*(e**2 + 2* c**2) - 4 * e **2) / denom
        elif alpha == float('-inf'):
            # Welsch/Leclerc loss
            first_term = ((e)**2 * np.exp( -((e**2)/(2* (c**2)))))/(c**4)
            second_term = np.exp( -((e**2)/(2* (c**2))))/(c**2)
            result = (first_term - second_term)
        else:
            # General case
            e2 = (e)**2
            denom = c**2 * np.abs(alpha - 2)
            A = e2 / denom + 1

            term1 = -2 * (alpha/2 - 1) * A**(alpha/2 - 2) * e2
            term1 /= (c**4 * np.abs(alpha - 2))

            term2 = - A**(alpha/2 - 1) / (c**2)

            result = term1 + term2
            
        # Return scalar if input was scalar
        if scalar_input:
            return result[0]
        return result
    
    def dpsi_dalpha(self, e_t, c, alpha_loss):
        # define a small neighborhood around the problematic points:
        eps = 1e-4

        # if α is within ±eps of 2, treat it as exactly 2 (L2 case)
        if abs(alpha_loss - 2) < eps:
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) 
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) 
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha

        # similarly guard α≈0
        if abs(alpha_loss - 0) < eps:
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) 
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) 
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha

        # if you ever parameterize "α = −∞" as, say, alpha_loss < some negative cutoff
        if alpha_loss < -1e3:
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) 
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c)
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha  

        # otherwise you're safely away from the singularities, so use your general formula:
        alpha = alpha_loss
        e2 = e_t**2
        denom = c**2 * np.abs(alpha - 2)
        A = (e2 / denom) + 1

        # Compute the bracket term
        term_log = np.log(A) / 2
        term_frac = (e2 * (alpha/2 - 1)) / (c**2 * A * np.abs(alpha - 2) * (alpha - 2))
        bracket = term_log - term_frac

        # Combine everything
        result = (e_t * A**(alpha/2 - 1) * bracket) / (c**2)
        return result

    def dpsi_dc(self, e_t, c, alpha_loss):
        eps = 1e-4

        if alpha_loss == 2:
            # dψ/dc = -2e / c³
            return -2 * e_t / c**3

        elif abs(alpha_loss) < eps:
            # dψ/dc = -8ec / (e² + 2c²)²
            return -8 * e_t * c / (e_t**2 + 2 * c**2)**2

        elif alpha_loss < -1e3:
            # dψ/dc = ( -2e / c³ + e³ / c⁵ ) * exp(-½ (e/c)²)
            z = e_t / c
            exp_term = np.exp(-0.5 * z**2)
            return (-2 * e_t / c**3 + e_t**3 / c**5) * exp_term

        else:
            alpha = alpha_loss
            e2 = e_t**2
            denom = np.abs(alpha - 2) * c**2
            A = e2 / denom + 1

            term1 = -2 * e_t * A**(alpha/2 - 1) / (c**3)
            term2 = -2 * e_t**3 * (alpha/2 - 1) * A**(alpha/2 - 2) / (np.abs(alpha - 2) * c**5)
            
            return term1 + term2
    
    def _filter_volatility(self, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        T = len(y)
        omega, gamma, beta = params[:3]
        
        param_idx = 3
        
        # Get alpha_loss parameter
        if self.alpha_loss is None:
            alpha_loss = params[param_idx]
            param_idx += 1
        else:
            alpha_loss = self.alpha_loss
        
        # Get c parameter
        if self.c is None:
            c = params[param_idx]
        else:
            c = self.c
        
        # Initialize volatility with the unconditional variance
        f = np.zeros(T+1)
        f[0] = omega/(1-beta)
        
        # Recursively update the volatility
        for t in range(T):
            e_t = y[t]**2 - f[t]
            psi_t = self._rho_derivative(e_t, alpha_loss, c)   
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            
            # Enforce positive volatility
            f[t+1] = max(f[t+1], 1e-12)
        
        return f[1:]
    
    def _compute_derivatives(self, y: np.ndarray, f: np.ndarray, params: np.ndarray) -> np.ndarray:
        T = len(y)
        omega, gamma, beta = params[:3]
        
        param_idx = 3
        n_params = 3
        
        # Get alpha_loss parameter
        if self.alpha_loss is None:
            alpha_loss = params[param_idx]
            param_idx += 1
            n_params += 1
        else:
            alpha_loss = self.alpha_loss
        
        # Get c parameter
        if self.c is None:
            c = params[param_idx]
            n_params += 1
        else:
            c = self.c
        
        # Initialize derivatives of f_t with respect to theta
        e0 = y[0]**2 - f[0]
        psi0 = self._rho_derivative(e0, alpha_loss, c)
        d2psi0_df = self._rho_second_derivative(e0, alpha_loss, c) 
        df_dtheta = np.zeros((T, n_params))
        df_dtheta[0, 0] = 1                  # ∂f₁/∂ω
        df_dtheta[0, 1] = psi0               # ∂f₁/∂γ
        df_dtheta[0, 2] = f[0]               # ∂f₁/∂β
        
        # Initialize derivatives for alpha_loss and c if they are estimated
        param_idx = 3
        
        if self.alpha_loss is None:
            dpsi0_dalpha = self.dpsi_dalpha(e0, c, alpha_loss)
            df_dtheta[0, param_idx] = gamma * dpsi0_dalpha  # ∂f₁/∂α
            param_idx += 1
        
        if self.c is None:
            dpsi0_dc = self.dpsi_dc(e0, c, alpha_loss)
            df_dtheta[0, param_idx] = gamma * dpsi0_dc  # ∂f₁/∂c
    
        # According to the recursive formula:
        for t in range(1, T):
            e_t = (y[t-1]**2 - f[t-1])
            psi_t = self._rho_derivative(e_t, alpha_loss, c)  
            d2psi_df = self._rho_second_derivative(e_t, alpha_loss, c)   
            
            # Common term in recursive updates
            common_term = gamma * d2psi_df + beta
            
            # Derivative with respect to parameters
            # For omega (∂ω/∂θ_0 = 1, else 0)
            df_dtheta[t, 0] = 1 + df_dtheta[t-1, 0] * common_term
            
            # For gamma (∂γ/∂θ_1 = 1, else 0)
            df_dtheta[t, 1] = psi_t + df_dtheta[t-1, 1] * common_term
            
            # For beta (∂β/∂θ_2 = 1, else 0)
            df_dtheta[t, 2] = f[t-1] + df_dtheta[t-1, 2] * common_term
            
            # Reset param_idx for additional parameters
            param_idx = 3
            
            # Derivative with respect to alpha_loss, if applicable
            if self.alpha_loss is None:
                dpsi_dalpha = self.dpsi_dalpha(e_t, c, alpha_loss)
                df_dtheta[t, param_idx] = gamma * dpsi_dalpha + common_term * df_dtheta[t-1, param_idx]
                param_idx += 1
            
            # Derivative with respect to c, if applicable
            if self.c is None:
                dpsi_dc = self.dpsi_dc(e_t, c, alpha_loss)
                df_dtheta[t, param_idx] = gamma * dpsi_dc + common_term * df_dtheta[t-1, param_idx]
        
        return df_dtheta
    
    def _qle_objective(self, params: np.ndarray, y: np.ndarray) -> float:
        try:
            # Apply parameter constraints
            param_idx = 3
            
            if self.alpha_loss is None:
                # Ensure alpha_loss is in reasonable range
                if params[param_idx] < -10 or params[param_idx] > 10:
                    return 1e10
                param_idx += 1
            
            if self.c is None:
                # Ensure c is positive and reasonable
                if params[param_idx] <= 0 or params[param_idx] > 10:
                    return 1e10
            
            # Basic parameter constraints for volatility model stability
            if params[0] <= 0 or params[1] < 0 or params[2] < 0 or params[2] >= 1 or params[1] + params[2] >= 0.999:
                return 1e10
            
            # Filter volatility
            f = self._filter_volatility(y, params)
            
            # Compute residuals - h_t is defined as y_t^k - f_t(θ)
            h_t = y**2 - f
            
            # Use f as the conditional variance approximation
            sigma2_t = f
            
            # Compute derivatives of f_t with respect to parameters
            df_dtheta = self._compute_derivatives(y, f, params)
            
            # G_t(θ) = (1/T) * sum[ h_t(θ) / σ²_t(θ) * ∂f_t(θ)/∂θ ]
            G_t = np.sum(h_t.reshape(-1, 1) / sigma2_t.reshape(-1, 1) * df_dtheta, axis=0) / len(y)
            
            # The objective is to minimize ||G_t(θ)||²
            obj = np.linalg.norm(G_t)
            
            return obj
        except Exception as e:
            # print(f"Error in objective function: {e}")
            return 1e10
    
    def fit(self, y: np.ndarray, initial_params: Optional[Dict] = None, 
            method: str = 'Nelder-Mead', maxiter: int = 2000, verbose: bool = False) -> Dict:
        
        # Set default initial parameters if not provided
        if initial_params is None:
            initial_params = {
                'omega': 0.07, 
                'gamma': 0.11, 
                'beta': 0.8
            }
            if self.alpha_loss is None:
                initial_params['alpha_loss'] = 1  # Default value close to Cauchy loss
            
            if self.c is None:
                initial_params['c'] = 1.2  # Default value for scale parameter
        
        # Prepare initial parameter array
        init_params = np.array([initial_params[name] for name in self.param_names])
        
        # Run optimization
        options = {'maxiter': maxiter, 'disp': verbose}
        
        # Different optimization methods may work better in different cases
        if method == 'Nelder-Mead':
            options['adaptive'] = True  # Use adaptive Nelder-Mead for better convergence
            result = minimize(
                self._qle_objective, 
                init_params, 
                args=(y,), 
                method=method, 
                options=options,
            )
        elif method == 'BFGS':
            options = {'maxiter': maxiter, 'gtol': 1e-6, 'disp': verbose}
            result = minimize(
                self._qle_objective, 
                init_params, 
                args=(y,), 
                method=method, 
                options=options,
            )
        
        if not result.success and method != 'differential_evolution' and verbose:
            print(f"Warning: Optimization did not converge: {result.message}")
            
            # Try again with different method if first one failed
            if method == 'Nelder-Mead':
                if verbose:
                    print("Trying BFGS method instead...")
                result = minimize(
                    self._qle_objective,
                    init_params,
                    args=(y,),
                    method='BFGS',
                    options={'maxiter': maxiter, 'disp': verbose}
                )
        
        # Store parameters
        self.params = {name: val for name, val in zip(self.param_names, result.x)}
        
        # Compute fitted volatility
        param_array = np.array([self.params[name] for name in self.param_names])
        self.fitted_volatility = self._filter_volatility(y, param_array)
        self.residuals = y**2 - self.fitted_volatility
        
        if verbose:
            print(f"Optimization result: {result.message}")
            print(f"Parameters: {self.params}")
        
        return self.params
    
    def simulate(self, T: int, dist: str = 't', df: int = 5, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.params is None:
            raise ValueError("Parameters must be set before simulation")
        
        if seed is not None:
            np.random.seed(seed)
        
        omega = self.params['omega']
        gamma = self.params['gamma']
        beta = self.params['beta']
        
        if self.alpha_loss is None:
            alpha_loss = self.params['alpha_loss']
        else:
            alpha_loss = self.alpha_loss
        
        if self.c is None:
            c = self.params['c']
        else:
            c = self.c
        
        # Initialize arrays
        y = np.zeros(T)
        f = np.zeros(T+1)
        f[0] = omega / (1 - beta)  # Start at unconditional variance
        
        # Generate innovations
        if dist == 't':
            eps = np.random.standard_t(df, T)
        else:
            eps = np.random.normal(0, 1, T)
            
        # Generate data
        for t in range(T):
            # Generate return
            y[t] = np.sqrt(f[t]) * eps[t]
            
            # Compute the score
            e_t = (y[t]**2 - f[t])
            psi_t = self._rho_derivative(e_t, alpha_loss, c) 
            
            # Update volatility
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            f[t+1] = max(f[t+1], 1e-12)  # Ensure positive volatility
            
        return y, f[1:]

def run_combined_monte_carlo_simulation(num_repetitions=500, sample_size=2000):
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
        'alpha_loss': 1,
        'c': 1.2
    }
    
    # Initialize arrays to store results
    fixed_alpha0_results = []
    fixed_alpha2_results = []  # New array for alpha=2
    estimated_results = []
    
    # Define column names for all result types
    fixed_columns = ['rep_id', 'omega', 'gamma', 'beta', 'c', 'convergence', 'rmse', 'runtime']
    estimated_columns = ['rep_id', 'omega', 'gamma', 'beta', 'alpha_loss', 'c', 'convergence', 'rmse', 'runtime']
    
    # Start simulation
    print(f"Starting Combined Monte Carlo simulation with {num_repetitions} repetitions")
    start_time_total = time.time()
    
    for i in range(num_repetitions):
        rep_seed = 42 + i  # Different seed for each repetition
        
        # Create simulation model with true parameters
        sim_model = RobustQLEVolatilityModel(alpha_loss=true_params['alpha_loss'])
        sim_model.params = true_params.copy()
        
        # Simulate data - to be used by all model types
        y, true_vol = sim_model.simulate(sample_size, dist='n', seed=rep_seed)
        
        # ---------- Fixed Alpha=0 Model ----------
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

        #calculate covergage probaility
        # if fixed_alpha0_convergence and fixed_alpha0_model.fitted_volatility is not None:
        #     eps = 1e-4
        #     upper_val = np.array(fixed_alpha0_model.params)+eps
        #     lower_val = np.array(fixed_alpha0_model.params)-eps
        #     upper = fixed_alpha0_model._qle_objective(upper, y)
        #     lower = fixed_alpha0_model._qle_objective(lower, y)
        #     deriv = (upper-lower)/(2*eps)

        #     lower_bound = fixed_alpha0_model.fitted_volatility - 1.96 * np.sqrt(fixed_alpha0_model.fitted_volatility) / np.sqrt(sample_size)
        #     upper_bound = fixed_alpha0_model.fitted_volatility + 1.96 * np.sqrt(fixed_alpha0_model.fitted_volatility) / np.sqrt(sample_size)
        #     if lower_bound < true_vol < upper_bound:
        #         fixed_alpha0_coverage = 1
        #     else:
        #         fixed_alpha0_coverage = 0

        
        # Calculate RMSE if convergence was successful
        if fixed_alpha0_convergence and fixed_alpha0_model.fitted_volatility is not None:
            fixed_alpha0_rmse = np.sqrt(np.mean((fixed_alpha0_model.fitted_volatility - true_vol)**2))
        else:
            fixed_alpha0_rmse = np.nan
        
        # Record results for fixed alpha=0 model
        fixed_alpha0_result = [i+1]  # rep_id
        
        if fixed_alpha0_convergence:
            for param in fixed_columns[1:-3]:  # Skip rep_id, convergence, rmse, runtime
                if param in fixed_alpha0_model.params:
                    fixed_alpha0_result.append(fixed_alpha0_model.params[param])
                else:
                    fixed_alpha0_result.append(np.nan)
        else:
            # If convergence failed, fill with NaNs
            fixed_alpha0_result.extend([np.nan] * (len(fixed_columns) - 3))
            
        fixed_alpha0_result.extend([fixed_alpha0_convergence, fixed_alpha0_rmse, fixed_alpha0_runtime])
        fixed_alpha0_results.append(fixed_alpha0_result)
        #fixed_alpha0_results.append(fixed_alpha0_coverage)
        
        # ---------- NEW: Fixed Alpha=2 Model ----------
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

        # if fixed_alpha2_convergence and fixed_alpha2_model.fitted_volatility is not None:
        #     eps = 1e-4
        #     upperval = np.array(fixed_alpha2_model.params)+eps
        #     lowerval = np.array(fixed_alpha2_model.params)-eps
        #     upper = fixed_alpha2_model._qle_objective(upperval, y)
        #     lower = fixed_alpha2_model._qle_objective(lowerval, y)
        #     deriv = (upper-lower)/(2*eps)

        #     lower_bound = fixed_alpha2_model.fitted_volatility - 1.96 * np.sqrt(fixed_alpha2_model.fitted_volatility) / np.sqrt(sample_size)
        #     upper_bound = fixed_alpha2_model.fitted_volatility + 1.96 * np.sqrt(fixed_alpha2_model.fitted_volatility) / np.sqrt(sample_size)
        #     if lower_bound < true_vol < upper_bound:
        #         fixed_alpha2_coverage = 1
        #     else:
        #         fixed_alpha2_coverage = 0
        
        # Calculate RMSE if convergence was successful
        if fixed_alpha2_convergence and fixed_alpha2_model.fitted_volatility is not None:
            fixed_alpha2_rmse = np.sqrt(np.mean((fixed_alpha2_model.fitted_volatility - true_vol)**2))
        else:
            fixed_alpha2_rmse = np.nan
        
        # Record results for fixed alpha=2 model
        fixed_alpha2_result = [i+1]  # rep_id
        
        if fixed_alpha2_convergence:
            for param in fixed_columns[1:-3]:  # Skip rep_id, convergence, rmse, runtime
                if param in fixed_alpha2_model.params:
                    fixed_alpha2_result.append(fixed_alpha2_model.params[param])
                else:
                    fixed_alpha2_result.append(np.nan)
        else:
            # If convergence failed, fill with NaNs
            fixed_alpha2_result.extend([np.nan] * (len(fixed_columns) - 3))
            
        fixed_alpha2_result.extend([fixed_alpha2_convergence, fixed_alpha2_rmse, fixed_alpha2_runtime])
        fixed_alpha2_results.append(fixed_alpha2_result)
        #ixed_alpha2_results.append(fixed_alpha2_coverage)
        
        # ---------- Estimated Alpha Model ----------
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

        # if est_convergence and estimated_model.fitted_volatility is not None:
        #     eps = 1e-4
        #     upper = estimated_model._qle_objective(estimated_model.params+eps, y)
        #     lower = estimated_model._qle_objective(estimated_model.params-eps, y)
        #     deriv = (upper-lower)/(2*eps)

        #     lower_bound = estimated_model.fitted_volatility - 1.96 * np.sqrt(estimated_model.fitted_volatility) / np.sqrt(sample_size)
        #     upper_bound = estimated_model.fitted_volatility + 1.96 * np.sqrt(estimated_model.fitted_volatility) / np.sqrt(sample_size)
        #     if lower_bound < true_vol < upper_bound:
        #         estimated_model_coverage = 1
        #     else:
        #         estimated_model_coverage = 0
        
        # Calculate RMSE if convergence was successful
        if est_convergence and estimated_model.fitted_volatility is not None:
            est_rmse = np.sqrt(np.mean((estimated_model.fitted_volatility - true_vol)**2))
        else:
            est_rmse = np.nan
        
        # Record results for estimated alpha model
        est_result = [i+1]  # rep_id
        
        if est_convergence:
            for param in estimated_columns[1:-3]:  # Skip rep_id, convergence, rmse, runtime
                if param in estimated_model.params:
                    est_result.append(estimated_model.params[param])
                else:
                    est_result.append(np.nan)
        else:
            # If convergence failed, fill with NaNs
            est_result.extend([np.nan] * (len(estimated_columns) - 3))
            
        est_result.extend([est_convergence, est_rmse, est_runtime])
        estimated_results.append(est_result)
        #estimated_results.append(estimated_model_coverage)

        print(i)

        # Print progress
        if (i+1) % 50 == 0 or i+1 == num_repetitions:
            elapsed = time.time() - start_time_total
            print(f"Completed {i+1}/{num_repetitions} repetitions. Time elapsed: {elapsed:.2f} seconds")
    
    # Convert results to DataFrames
    fixed_alpha0_df = pd.DataFrame(fixed_alpha0_results, columns=fixed_columns)
    fixed_alpha2_df = pd.DataFrame(fixed_alpha2_results, columns=fixed_columns)
    estimated_df = pd.DataFrame(estimated_results, columns=estimated_columns)


    return fixed_alpha0_df, fixed_alpha2_df, estimated_df


if __name__ == "__main__":
    # Run combined MC simulation
    fixed_alpha0_results, fixed_alpha2_results, estimated_results = run_combined_monte_carlo_simulation(
        num_repetitions=500, sample_size=4000
    )
    
    # Save results to CSV files
    fixed_alpha0_results.to_csv('monte_carlo_fixed_alpha0_results_final_4000.csv', index=False)
    fixed_alpha2_results.to_csv('monte_carlo_fixed_alpha2_results_final_4000.csv', index=False)
    estimated_results.to_csv('monte_carlo_estimated_alpha_results_final_4000.csv', index=False)
    
    # Additional analysis - save paired comparison
    common_reps = np.intersect1d(
        np.intersect1d(
            fixed_alpha0_results.loc[fixed_alpha0_results['convergence'], 'rep_id'],
            fixed_alpha2_results.loc[fixed_alpha2_results['convergence'], 'rep_id']
        ),
        estimated_results.loc[estimated_results['convergence'], 'rep_id']
    )
    