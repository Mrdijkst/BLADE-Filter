import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, Optional, Union
import autograd 
from autograd import grad, jacobian
import warnings
warnings.filterwarnings('ignore')


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
            result = np.ones_like(e) / (c**2)
        elif alpha == 0:
            # Cauchy/Lorentzian loss
            denom = (1 + 0.5 * (e/c)**2)**2
            result = (1 / c**2) * (1 - 0.5 * (e/c)**2) / denom
        elif alpha == float('-inf'):
            # Welsch/Leclerc loss
            result = (1 / c**2) * (1 - (e**2 / c**2)) * np.exp(-0.5 * (e/c)**2)
        else:
            # General case
            base_term = np.power((e/c)**2 / np.abs(alpha-2) + 1, alpha/2 - 2)
            part1 = (1 / c**2) * base_term
            part2 = (e/c)**2 / np.abs(alpha-2) + 1
            part3 = part2 + (alpha/2 - 1) * (2 * e**2) / (c**2 * np.abs(alpha-2))
            result = part1 * part3
            
        
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
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) * (-1)
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) * (-1)
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha  # ∂ψ/∂α = 0 for the L2 loss

        # similarly guard α≈0
        if abs(alpha_loss - 0) < eps:
            # use the closed-form ∂ψ/∂α at α=0 (which you've already set to 0)
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) * (-1)
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) * (-1)
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha

        # if you ever parameterize "α = −∞" as, say, alpha_loss < some negative cutoff
        if alpha_loss < -1e3:
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) * (-1)
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) * (-1)
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha  

        # otherwise you're safely away from the singularities, so use your general formula:
        abs_alpha_minus_2 = np.abs(alpha_loss - 2)
        denominator_inner = c**2 * abs_alpha_minus_2
        inner_term = (e_t**2) / denominator_inner + 1
        power_term = inner_term**(alpha_loss / 2 - 1)

        log_term = np.log(inner_term) / 2
        frac_term = (e_t**2 * (alpha_loss / 2 - 1)) / (
            c**2 * inner_term * abs_alpha_minus_2 * (alpha_loss - 2)
        )

        numerator = e_t * power_term * (log_term - frac_term)

        return (numerator / c**2)
    
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
            # General case:
            # dψ/dc = [ -2e / c³ 
            #           + (e / c²)(α/2 - 1)( (e² / (c²|α−2|) + 1 )**(α/2 - 2) ) * (-2e² / (c³|α−2|)) ]
            #         * (e² / (c²|α−2|) + 1 )**(α/2 - 1)
            abs_alpha_diff = np.abs(alpha_loss - 2)
            z_sq = (e_t**2) / (c**2 * abs_alpha_diff)
            base = z_sq + 1
            power1 = (alpha_loss / 2) - 2
            power2 = (alpha_loss / 2) - 1

            first_term = -2 * e_t / c**3
            second_term = (e_t / c**2) * ((alpha_loss / 2) - 1) * (base**power1) * (-2 * e_t**2 / (c**3 * abs_alpha_diff))

            return (first_term + second_term) * base**power2
    
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
            psi_t = self._rho_derivative(e_t, alpha_loss, c)   # -1 because e_t = y_t^2 - f_t
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
        psi0 =  self._rho_derivative(e0, alpha_loss, c)
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
            psi_t = self._rho_derivative(e_t, alpha_loss, c)  # -1 because e_t = y_t^2 - f_t
            d2psi_df = self._rho_second_derivative(e_t, alpha_loss, c)   # Chain rule
            
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
            print(f"Error in objective function: {e}")
            return 1e10
    
    def fit(self, y: np.ndarray, initial_params: Optional[Dict] = None, 
            method: str = 'Nelder-Mead', maxiter: int = 2000) -> Dict:
        
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
                initial_params['c'] = 1.1  # Default value for scale parameter
        
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
            
            # Basic parameters bounds
            bounds.extend([(0.001, 0.5), (0.001, 0.5), (0.6, 0.999)])
            
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
        
        # Compute fitted volatility
        param_array = np.array([self.params[name] for name in self.param_names])
        self.fitted_volatility = self._filter_volatility(y, param_array)
        self.residuals = y**2 - self.fitted_volatility
        
        print(f"Optimization result: {result.message}")
        print(f"Parameters: {self.params}")
        
        return self.params
    
    
    
    def plot_volatility(self, y: np.ndarray, true_vol: np.ndarray = None, title: str = "Estimated Volatility") -> None:
        
        if self.fitted_volatility is None:
            raise ValueError("Model must be fit before plotting")
        
        plt.figure(figsize=(12, 8))
        
        # Plot original data
        plt.subplot(2, 1, 1)
        plt.plot(y, 'b-', alpha=0.5, label='Returns')
        plt.title(title)
        plt.legend()
        
        # Plot volatility
        plt.subplot(2, 1, 2)
        plt.plot(np.sqrt(self.fitted_volatility), 'r-', label='Estimated Volatility')
        
        if true_vol is not None:
            plt.plot(np.sqrt(true_vol), 'b-', alpha=0.3, label='True Volatility')
        
        plt.legend()
        plt.ylim(0, np.max(np.sqrt(self.fitted_volatility)) * 1.5)
        plt.tight_layout()
        plt.show()
    
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
            psi_t = self._rho_derivative(e_t, alpha_loss, c) * (1)
            
            # Update volatility
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            f[t+1] = max(f[t+1], 1e-12)  # Ensure positive volatility
            
        return y, f[1:]


def demo_simulated_data():
    """Demonstrate the model with simulated data"""
    # Set parameters for the data generating process
    true_params = {
        'omega': 0.07,
        'gamma': 0.11,
        'beta': 0.80,
        'alpha_loss': 1,
        'c': 1.1    # Cauchy loss
    }
    
    # Create a model with the true parameters for simulation
    sim_model = RobustQLEVolatilityModel(alpha_loss=true_params['alpha_loss'])
    sim_model.params = true_params
    
    # Simulate data
    T = 3000
    y, true_vol = sim_model.simulate(T, dist='t', df=5, seed=41)
    
    # Create a model for estimation
    # Option 1: Fix alpha to a known value
    model1 = RobustQLEVolatilityModel(alpha_loss=0)  # Fix to Cauchy loss
    
    # Option 2: Estimate alpha along with other parameters
    model2 = RobustQLEVolatilityModel()  # Alpha will be estimated
    
    # Fit the models
    print("\nFitting model with fixed alpha=0 (Cauchy loss):")
    params1 = model1.fit(y, method='Nelder-Mead')
    
    print("\nFitting model with estimated alpha:")
    params2 = model2.fit(y, method='Nelder-Mead')
    
    # Compare with true parameters
    print("\nTrue parameters:")
    for key, val in true_params.items():
        print(f"  {key}: {val:.4f}")
    
    print("\nEstimated parameters (fixed alpha):")
    for key, val in params1.items():
        print(f"  {key}: {val:.4f}")
    
    print("\nEstimated parameters (estimated alpha):")
    for key, val in params2.items():
        print(f"  {key}: {val:.4f}")
    
    # Plot results
    model1.plot_volatility(y, true_vol=true_vol, title="QLE Volatility Model with Fixed Alpha")
    model2.plot_volatility(y,true_vol=true_vol, title="QLE Volatility Model with Estimated Alpha")

    # Compare RMSE of volatility estimation
    rmse1 = np.sqrt(np.mean((model1.fitted_volatility[T-2000:T] - true_vol[T-2000:T])**2))
    rmse2 = np.sqrt(np.mean((model2.fitted_volatility[T-2000:T] - true_vol[T-2000:T])**2))
    
    print(f"\nRMSE of volatility estimation (fixed alpha): {rmse1:.6f}")
    print(f"RMSE of volatility estimation (estimated alpha): {rmse2:.6f}")
    
    return model1, model2, y, true_vol



if __name__ == "__main__":
    # Demo with simulated data
    model1, model2, sim_y, true_vol = demo_simulated_data()
    
    #notes: When using e_t= (y_t^2 - f_t)**2 instead of e_t= (y_t - f_t), the estimation of the parameters are worse.
    # but the volatility estimation is better.

"notes: When \alpha=1, then fixing \alpha to 1 results in a good estimation. BUt it we don't do this then it does not work."


#grid search if needed


def grid_search_alpha_c(alpha_values=None, c_values=None):
    
    # Set parameters for the data generating process
    true_params = {
        'omega': 0.05,
        'gamma': 0.15,
        'beta': 0.8,
        'alpha_loss': 1,
        'c': 1.5
    }
    
    # Create a model with the true parameters for simulation
    sim_model = RobustQLEVolatilityModel(alpha_loss=true_params['alpha_loss'], c=true_params['c'])
    sim_model.params = true_params
    
    # Simulate data
    T = 3000
    y, true_vol = sim_model.simulate(T, dist='t', df=5, seed=41)

    # If both alpha and c are provided as fixed values, just fit the model
    if alpha_values is not None and not isinstance(alpha_values, list) and c_values is not None and not isinstance(c_values, list):
        model = RobustQLEVolatilityModel(alpha_loss=alpha_values, c=c_values)
        return model.fit(y), None, None

    # Set default grid values if not provided
    if alpha_values is None:
        alpha_values = np.concatenate([
            np.array([-2, -1, -0.5]),  # More robust than Cauchy
            np.linspace(0, 2, 11),     # Cauchy to L2 range (most common)
            np.array([3, 4])           # Less robust than L2
        ])
    
    if c_values is None:
        c_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    
    best_alpha = None
    best_c = None
    best_nll = float('inf')
    best_params = None

    # Perform grid search
    results = []
    
    for alpha in alpha_values:
        for c in c_values:
            temp_model = RobustQLEVolatilityModel(alpha_loss=alpha, c=c)
            
            try:
                temp_params = temp_model.fit(y)
                nll = temp_model._qle_objective(np.array(list(temp_params.values())), y)
                
                results.append({
                    'alpha': alpha,
                    'c': c,
                    'nll': nll,
                    'params': temp_params
                })
                
                if nll < best_nll:
                    best_nll = nll
                    best_alpha = alpha
                    best_c = c
                    best_params = temp_params
                
                print(f"Alpha: {alpha}, c: {c}, NLL: {nll:.4f}")
            except Exception as e:
                print(f"Error with alpha={alpha}, c={c}: {e}")
    
    #
