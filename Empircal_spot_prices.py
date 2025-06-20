import pandas as pd
#from Location_model_QLE3 import RobustQLELocationModel
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class RobustQLELocationModel:
    
    def __init__(self, alpha_loss: float = None, c: float = None):
        
        self.alpha_loss = alpha_loss
        self.c = c
        self.params = None
        self.param_names = ['omega', 'gamma', 'beta']
        if alpha_loss is None:
            self.param_names.append('alpha_loss')
        if c is None:
            self.param_names.append('c')
        self.fitted_location = None
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

            return term1 + term2
            
        
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
            return dpsi_dalpha  # ∂ψ/∂α = 0 for the L2 loss

        # similarly guard α≈0
        if abs(alpha_loss - 0) < eps:
            # use the closed-form ∂ψ/∂α at α=0 (which you've already set to 0)
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
        alpha= alpha_loss
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
            # General case:
            
    
    def _filter_location(self, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        
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
        
        # Initialize location with the unconditional mean
        f = np.zeros(T+1)
        f[0] = y[0]
        
        # Recursively update the location
        for t in range(T):
            # Use e(y_t, f_t) = y_t - f_t for location model
            e_t = (y[t] - f[t])
            psi_t = self._rho_derivative(e_t, alpha_loss, c)   #
            f[t+1] = omega + gamma * psi_t + beta * f[t]
        
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
        e0 = y[0] - f[0]
        psi0 = self._rho_derivative(e0, alpha_loss, c) * (1)
        d2psi0_df = self._rho_second_derivative(e0, alpha_loss, c) * (-1)**2  # Chain rule
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
            e_t = (y[t-1] - f[t-1])
            psi_t = self._rho_derivative(e_t, alpha_loss, c)   # Negative because e_t = y_t - f_t
            d2psi_df = self._rho_second_derivative(e_t, alpha_loss, c)  # Chain rule
            
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
            
            # Basic parameter constraints for location model stability
            if params[0] <= -10 or params[0] >= 10 or params[1] < 0 or params[2] < 0 or params[2] >= 1 or params[1] + params[2] >= 0.999:
                return 1e10
            
            # Filter location
            f = self._filter_location(y, params)
            
            # Compute residuals - h_t is defined as y_t - f_t(θ)
            h_t = y - f
            
            # Use constant variance for the location model
            sigma2_t = np.ones_like(f)
            
            # Compute derivatives of f_t with respect to parameters
            df_dtheta = self._compute_derivatives(y, f, params)
            
            # G_t(θ) = (1/T) * sum[ h_t(θ) / σ²_t(θ) * ∂f_t(θ)/∂θ ]
            G_t = np.sum(h_t.reshape(-1, 1) / sigma2_t.reshape(-1, 1) * df_dtheta, axis=0) / len(y)
            
            # The objective is to minimize ||G_t(θ)||²
            obj = np.linalg.norm(G_t)
            
            return obj
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e10
    
    def fit(self, y: np.ndarray, initial_params: Optional[Dict] = None, 
            method: str = 'Nelder-Mead', maxiter: int = 2000) -> Dict:
        


        # y_mean = np.mean(y)
        # y_std = np.std(y)
        # y_var = np.var(y)
        
        # # Estimate persistence using AR(1) model
        # y_lag = y[:-1]
        # y_curr = y[1:]
        
        # # Simple AR(1): y_t = c + φ*y_{t-1} + ε_t
        # # Using least squares to get rough estimates
        # X = np.column_stack([np.ones(len(y_lag)), y_lag])


        # ar_coeffs = np.linalg.lstsq(X, y_curr, rcond=None)[0]
        # ar_const, ar_phi = ar_coeffs
        
        # # Convert AR(1) to location model parameters
        # # For location model: f_t = ω + γ*ψ_t + β*f_{t-1}
        # # If we assume ψ_t ≈ (y_t - f_t), then roughly similar to AR(1)
        
        # # Beta (persistence): use AR(1) coefficient but constrain
        # beta_init = max(0.6, min(0.95, abs(ar_phi)))
        
        # # Omega (intercept): adjust for mean-reverting nature
        # omega_init = ar_const * (1 - beta_init)  # Unconditional mean adjustment
        # omega_init = max(-1.0, min(1.0, omega_init))  # Keep reasonable
        
        # # Gamma (sensitivity to innovations): 
        # # Start with moderate value, scaled by data variability
        # gamma_init = min(0.3, 0.1 * y_std)  # Scale with data volatility
        # gamma_init = max(0.01, gamma_init)   # Ensure positive
        
        # initial_params = {
        #     'omega': omega_init,
        #     'gamma': gamma_init, 
        #     'beta': beta_init
        # }
        
        # # Alpha_loss initialization
        # if self.alpha_loss is None:
        #     # For heavy-tailed financial data, start closer to Cauchy (α=0)
        #     # But not exactly 0 to avoid numerical issues
        #     initial_params['alpha_loss'] = 0.5
        
        # # C parameter initialization  
        # if self.c is None:
        #     # Use robust scale estimate (MAD - Median Absolute Deviation)
        #     residuals = y_curr - ar_const - ar_phi * y_lag
        #     mad = np.median(np.abs(residuals - np.median(residuals)))
        #     c_init = mad * 1.4826  # Convert MAD to std-like scale
        #     c_init = max(0.1, min(3.0, c_init))  # Keep in reasonable range
        #     initial_params['c'] = 0.5
        
        # print(f"Smart initial parameters:")
        # print(f"  Data characteristics: mean={y_mean:.3f}, std={y_std:.3f}")
        # print(f"  AR(1) estimates: const={ar_const:.3f}, phi={ar_phi:.3f}")
        # print(f"  Initial params: {initial_params}")
        
        
        # Set default initial parameters if not provided
        if initial_params is None:
            # Use sample mean for omega initialization
            mean_y = np.mean(y)
            initial_params = {
                'omega': 0.05,  # Start with a fraction of the mean
                'gamma': 0.2,    # adjust more realistically
                'beta': 0.75
            }
            if self.alpha_loss is None:
                initial_params['alpha_loss'] = 1  # Default value close to Cauchy loss
            
            if self.c is None:
                initial_params['c'] = 1.2  # Use standard deviation for scale parameter
        
        # Prepare initial parameter array
        init_params = np.array([initial_params[name] for name in self.param_names])
        
        # Run optimization
        options = {'maxiter': maxiter, 'disp': True}
        
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
            options = {'maxiter': maxiter, 'gtol': 1e-6}
            result = minimize(
                self._qle_objective, 
                init_params, 
                args=(y,), 
                method=method, 
                options=options,
            )
        elif method == 'differential_evolution':
            # Set up bounds for differential evolution
            bounds = []
            
            # Basic parameters bounds - adjusted for location model
            bounds.extend([(-2, 2), (0.001, 0.5), (0.6, 0.999)])
            
            # Alpha bounds if estimated
            if self.alpha_loss is None:
                bounds.append((-5, 5))
            
            # c bounds if estimated
            if self.c is None:
                bounds.append((0.1, 5.0))
            
            result = differential_evolution(
                self._qle_objective,
                bounds=bounds,
                args=(y,),
                strategy='best1bin',
                maxiter=maxiter,
                disp=True,
                polish=True
            )
        
        if not result.success and method != 'differential_evolution':
            print(f"Warning: Optimization did not converge: {result.message}")
            
            # Try again with different method if first one failed
            if method == 'Nelder-Mead':
                print("Trying BFGS method instead...")
                result = minimize(
                    self._qle_objective,
                    init_params,
                    args=(y,),
                    method='BFGS',
                    options={'maxiter': maxiter}
                )
        
        # Store parameters
        self.params = {name: val for name, val in zip(self.param_names, result.x)}
        
        # Compute fitted location
        param_array = np.array([self.params[name] for name in self.param_names])
        self.fitted_location = self._filter_location(y, param_array)
        self.residuals = y - self.fitted_location
        
        print(f"Optimization result: {result.message}")
        print(f"Parameters: {self.params}")
        
        return self.params
    
    def plot_location(self, y: np.ndarray, true_loc: np.ndarray = None, title: str = "Estimated Location") -> None:
        
        if self.fitted_location is None:
            raise ValueError("Model must be fit before plotting")
        
        plt.figure(figsize=(12, 6))
        
        # Plot original data and estimated location
        
        plt.plot(self.fitted_location, 'r-', linewidth=2, label='Estimated Location')
        
        if true_loc is not None:
            plt.plot(true_loc, 'g--', alpha=0.8, label='True Location')
        
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # def simulate(self, T: int, dist: str = 't', df: int = 5, noise_scale: float = 1.0, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        
    #     if self.params is None:
    #         raise ValueError("Parameters must be set before simulation")
        
    #     if seed is not None:
    #         np.random.seed(seed)
        
    #     omega = self.params['omega']
    #     gamma = self.params['gamma']
    #     beta = self.params['beta']
        
    #     if self.alpha_loss is None:
    #         alpha_loss = self.params['alpha_loss']
    #     else:
    #         alpha_loss = self.alpha_loss
        
    #     if self.c is None:
    #         c = self.params['c']
    #     else:
    #         c = self.c
        
    #     # Initialize arrays
    #     y = np.zeros(T)
    #     f = np.zeros(T+1)
    #     f[0] = omega / (1 - beta)  # Start at unconditional mean
        
    #     # Generate innovations
    #     if dist == 't':
    #         eps = np.random.standard_t(df, T) * noise_scale
    #     else:
    #         eps = np.random.normal(0, noise_scale, T)
        
    #     # Generate data
    #     for t in range(T):
    #         # Generate observation with noise
    #         y[t] = f[t] + eps[t]
            
    #         # Compute the score
    #         e_t = (y[t] - f[t])
    #         psi_t = self._rho_derivative(e_t, alpha_loss, c) * (1)
            
    #         # Update location
    #         f[t+1] = omega + gamma * psi_t + beta * f[t]
        
    #     return y, f[1:]

df  = pd.read_csv('/Users/MathijsDijkstra/Downloads/Elspotprices.csv', sep=';')

df['HourDK'] = pd.to_datetime(df['HourDK'], errors='coerce')
df['SpotPriceDKK'] = pd.to_numeric(df['SpotPriceDKK'].astype(str).str.replace(',', '.'), errors='coerce')
df['SpotPriceEUR'] = pd.to_numeric(df['SpotPriceEUR'].astype(str).str.replace(',', '.'), errors='coerce')
#df = df.dropna(subset=['HourDK', 'SpotPriceEUR'])
df = df.set_index('HourDK')
df = df[df.index < '2021-01-01']
#df = df[df['PriceArea'] == 'DK1
#plt.plot(df.index, df['SpotPriceEUR'], label='Spot Price EUR')
#plt.show()

print(len(df['SpotPriceEUR']))

# ...existing code...

df['DayOfWeek'] = df.index.dayofweek
df['Date'] = df.index.date

# Calculate daily averages for ALL days (not just Mondays)
daily_prices = df.groupby('Date')['SpotPriceEUR'].mean()
daily_prices.index = pd.to_datetime(daily_prices.index)

#take logs 
daily_prices = np.log(daily_prices)
print(f"Daily averages: {len(daily_prices)} observations")
print(f"Daily price statistics - Min: {daily_prices.min():.2f} DKK, Max: {daily_prices.max():.2f} DKK, Mean: {daily_prices.mean():.2f} DKK")



y = daily_prices.values
y= daily_prices.values- np.mean(daily_prices.values)

y = y/ np.std(daily_prices.values)
#y  =  np.log(df['SpotPriceEUR'].values)

# remove NaN values
# y = y[~np.isnan(y)]
# y = y[~np.isinf(y)]  # Remove infinite values if any
print(len(y))
model = RobustQLELocationModel( alpha_loss=1, c=1.2)  # Initialize model with alpha_loss and c


params = model.fit(y, method='Nelder-Mead')

# Convert params dict to a numpy array in the correct order
param_names = ['omega', 'gamma', 'beta']
if model.alpha_loss is None:
    param_names.append('alpha_loss')
if model.c is None:
    param_names.append('c')
params_array = np.array([params[name] for name in param_names])

# Get the estimated location (filtered values)
estimated_location = model._filter_location(y, params_array)


# Remove NaN/Inf from daily_prices before extracting y and index
mask = (~np.isnan(daily_prices.values)) & (~np.isinf(daily_prices.values))
filtered_prices = daily_prices[mask]
y = filtered_prices.values

plt.figure(figsize=(12, 6))
plt.plot(filtered_prices.index, y, label='Daily Average Spot Price', color='gray', alpha=0.7)
plt.plot(filtered_prices.index, estimated_location, label='Estimated Location', color='red', linewidth=2)
plt.title("Daily Spot Prices and Estimated Location")
plt.xlabel("Date")
plt.ylabel("Spot Price (EUR)")
plt.legend()
plt.tight_layout()
plt.show()