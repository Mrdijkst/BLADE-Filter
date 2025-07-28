
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
    """
    Implementation of the Quasi-Likelihood Estimation (QLE) volatility model with
    robust loss function as proposed by Barron (2019) in the context of 
    Quasi-Score Driven models.
    """
    
    def __init__(self, alpha_loss: float = None, c: float = 1):
        """
        Initialize the QLE volatility model with robust loss function.
        
        Parameters:
        -----------
        alpha_loss : float or None
            Robustness parameter for the loss function. Special cases:
            - alpha = 2: L2 loss (least squares)
            - alpha = 0: Cauchy/Lorentzian loss (log-based)
            - alpha = -∞: Welsch/Leclerc loss (exponential)
            If None, alpha will be estimated along with other parameters.
        c : float
            Scale parameter for the loss function.
        """
        self.alpha_loss = alpha_loss
        self.c = c
        self.params = None
        self.param_names = ['omega', 'gamma', 'beta']
        if alpha_loss is None:
            self.param_names.append('alpha_loss')
        self.fitted_volatility = None
        self.residuals = None
    
    def _rho_derivative(self, e: np.ndarray, alpha: float, c: float) -> np.ndarray:
        """
        First derivative of the robust loss function with respect to e.
        
        Parameters:
        -----------
        e : np.ndarray
            Error term, e(y_t, f_t) = y_t^2 - f_t for volatility model
        alpha : float
            Robustness parameter
        c : float
            Scale parameter
        
        Returns:
        --------
        np.ndarray
            First derivative of the loss function
        """
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
        """
        Second derivative of the robust loss function with respect to e.
        
        Parameters:
        -----------
        e : np.ndarray
            Error term
        alpha : float
            Robustness parameter
        c : float
            Scale parameter
        
        Returns:
        --------
        np.ndarray
            Second derivative of the loss function
        """
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
        if alpha_loss ==2:
            #Compute ∂ψ_t/∂α using numerical differentiation
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) * (-1)
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) * (-1)
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha
            return 0
        elif alpha_loss == 0:
            # #Compute ∂ψ_t/∂α using numerical differentiation
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) * (-1)
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) * (-1)
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha
            return 0
        elif alpha_loss == float('-inf'):
            #Compute ∂ψ_t/∂α using numerical differentiation
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) * (-1)
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) * (-1)
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha  
            return 0
        else:
            abs_alpha_minus_2 = np.abs(alpha_loss - 2)
            denominator_inner = c**2 * abs_alpha_minus_2
            inner_term = (e_t**2) / denominator_inner + 1
            power_term = inner_term**(alpha_loss / 2 - 1)
            
            log_term = np.log(inner_term) / 2
            frac_term = (e_t**2 * (alpha_loss / 2 - 1)) / (c**2 * inner_term * abs_alpha_minus_2 * (alpha_loss - 2))
            
            numerator = e_t * power_term * (log_term - frac_term)
            
            result = numerator / c**2
            
            return result
    #def dpsi_dalpha(self, e_t, c, alpha_loss):
        # define a small neighborhood around the problematic points:
        eps = 1e-4

        # if α is within ±eps of 2, treat it as exactly 2 (L2 case)
        if abs(alpha_loss - 2) < eps:
            return 0.0   # ∂ψ/∂α = 0 for the L2 loss

        # similarly guard α≈0
        if abs(alpha_loss - 0) < eps:
            # use the closed‐form ∂ψ/∂α at α=0 (which you’ve already set to 0)
            return 0.0

        # if you ever parameterize “α = −∞” as, say, alpha_loss < some negative cutoff
        if alpha_loss < -1e3:
            return 0.0

        # otherwise you’re safely away from the singularities, so use your general formula:
        abs_alpha_minus_2 = np.abs(alpha_loss - 2)
        denominator_inner = c**2 * abs_alpha_minus_2
        inner_term = (e_t**2) / denominator_inner + 1
        power_term = inner_term**(alpha_loss / 2 - 1)

        log_term = np.log(inner_term) / 2
        frac_term = (e_t**2 * (alpha_loss / 2 - 1)) / (
            c**2 * inner_term * abs_alpha_minus_2 * (alpha_loss - 2)
        )

        numerator = e_t * power_term * (log_term - frac_term)
        return numerator / c**2

    def _filter_volatility(self, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Filter the volatility process given the parameters and observations.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed time series
        params : np.ndarray
            Model parameters [omega, gamma, beta, (alpha_loss)]
        
        Returns:
        --------
        np.ndarray
            Filtered volatility (f_t)
        """
        T = len(y)
        omega, gamma, beta = params[:3]
        
        if self.alpha_loss is None:
            alpha_loss = params[3]
        else:
            alpha_loss = self.alpha_loss
        
        # Initialize volatility with the mean of y²
        f = np.zeros(T+1)
        #f[0] = np.mean(y[:200]**2) if T >= 20 else np.mean(y**2)
        f[0] = omega/(1-beta)
        
        # Recursively update the volatility
        for t in range(T):
            e_t = -(y[t]**2 - f[t])
            psi_t = self._rho_derivative(e_t, alpha_loss, self.c) * (-1)  # -1 because e_t = y_t^2 - f_t
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            
            # Enforce positive volatility
            f[t+1] = max(f[t+1], 1e-12)
        
        return f[1:]
    
    def _compute_derivatives(self, y: np.ndarray, f: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute the derivatives of f_t with respect to parameters.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed time series
        f : np.ndarray
            Filtered volatility
        params : np.ndarray
            Model parameters [omega, gamma, beta, (alpha_loss)]
        
        Returns:
        --------
        np.ndarray
            Derivatives of f_t with respect to parameters
        """
        T = len(y)
        omega, gamma, beta = params[:3]
        
        if self.alpha_loss is None:
            alpha_loss = params[3]
            n_params = 4
        else:
            alpha_loss = self.alpha_loss
            n_params = 3
        
        # Initialize derivatives of f_t with respect to theta
        e0 = y[0]**2 - f[0]
        psi0 = - self._rho_derivative(e0, alpha_loss, self.c)
        d2psi0_df =    self._rho_second_derivative(e0, alpha_loss, self.c) * (+1)
        df_dtheta = np.zeros((T, n_params))
        df_dtheta[0, 0] = 1                                # ∂f₁/∂ω
        df_dtheta[0, 1] =    psi0                          # ∂f₁/∂γ
        df_dtheta[0, 2] =    f[0]
    
        
        # According to the recursive formula:
        # ∂f_{t+1}(θ)/∂θ = ∂ω/∂θ + ψ_t*∂γ/∂θ + γ*∂ψ_t/∂θ + f_t*∂β/∂θ + (γ*∂ψ_t/∂f_t + β)*∂f_t/∂θ
        
        for t in range(1, T):
            e_t = -(y[t-1]**2 - f[t-1])
            psi_t = self._rho_derivative(e_t, alpha_loss, self.c) * (-1)  # -1 because e_t = y_t^2 - f_t
            d2psi_df = self._rho_second_derivative(e_t, alpha_loss, self.c) * (-1)**2  # Chain rule, (-1) for ∂e/∂f, squared for chain rule
            
            # Derivative with respect to parameters
            # For omega (∂ω/∂θ_0 = 1, else 0)
            df_dtheta[t, 0] = 1 +  df_dtheta[t-1, 0] *(gamma * d2psi_df + beta)
            
            # For gamma (∂γ/∂θ_1 = 1, else 0)
            df_dtheta[t, 1] = psi_t + df_dtheta[t-1, 1] * (gamma * d2psi_df + beta)
            
            # For beta (∂β/∂θ_2 = 1, else 0)
            df_dtheta[t, 2] = f[t-1] +  df_dtheta[t-1, 2] * (gamma * d2psi_df + beta)
            
            # Derivative with respect to alpha_loss, if applicable
            if self.alpha_loss is None:
                # Compute ∂ψ_t/∂α using numerical differentiation
                # delta = 1e-5
                # psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) * (-1)
                # psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) * (-1)
                # dpsi_dalpha1 = (psi_plus - psi_minus) / (2 * delta)
                
                dpsi_dalpha = self.dpsi_dalpha(e_t, self.c, alpha_loss)       
                
                
                # For alpha_loss (∂α/∂θ_3 = 1, else 0)
                df_dtheta[t, 3] = gamma * dpsi_dalpha + (beta+ d2psi_df*gamma) * df_dtheta[t-1, 3]
        
        return df_dtheta
    
    def _qle_objective(self, params: np.ndarray, y: np.ndarray) -> float:
        """
        Quasi-likelihood objective function to be minimized.
        
        Parameters:
        -----------
        params : np.ndarray
            Model parameters [omega, gamma, beta, (alpha_loss)]
        y : np.ndarray
            Observed time series
        
        Returns:
        --------
        float
            Objective function value
        """
        try:
            # Apply parameter constraints
            if self.alpha_loss is None and len(params) > 3:
                # Ensure alpha_loss is in reasonable range
                if params[3] < -10 or params[3] > 10:
                    return 1e10
            
            # Basic parameter constraints
            if params[0] <= 0 or params[1] < 0 or params[2] < 0 or params[2] >= 1 or params[1] + params[2] >= 0.999:
                return 1e10
            
            # Filter volatility
            f = self._filter_volatility(y, params)
            
            # Compute residuals - h_t is defined in the thesis as y_t^k - f_t(θ)
            # For volatility model, k=2, so h_t = y_t^2 - f_t
            h_t = y**2 - f
            
            # Use f as the conditional variance approximation
            # In the volatility model, σ²_t = f_t as mentioned in the thesis
            sigma2_t = f
            
            # Compute derivatives of f_t with respect to parameters
            df_dtheta = self._compute_derivatives(y, f, params)
            
            
            # G_t(θ) = (1/T) * sum[ h_t(θ) / σ²_t(θ) * ∂f_t(θ)/∂θ ]
            G_t = np.sum(h_t.reshape(-1, 1) / sigma2_t.reshape(-1, 1) * df_dtheta, axis=0) / len(y)
            
            # The objective is to minimize ||G_t(θ)||²
            obj = np.linalg.norm(G_t)
            #obj = np.sum(G_t**2)
            
            return obj
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e10
    
    def fit(self, y: np.ndarray, initial_params: Optional[Dict] = None, 
            method: str = 'Nelder-Mead', maxiter: int = 2000) -> Dict:
        """
        Fit the volatility model using QLE.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed time series
        initial_params : Dict, optional
            Initial parameter values. If None, default values will be used.
        method : str
            Optimization method for scipy.optimize.minimize
        maxiter : int
            Maximum number of iterations for optimization
        
        Returns:
        --------
        Dict
            Dictionary with fitted parameters
        """
        # Set default initial parameters if not provided
        if initial_params is None:
            initial_params = {
                'omega': 0.07, 
                'gamma': 0.11, 
                'beta': 0.8
            }
            if self.alpha_loss is None:
                initial_params['alpha_loss'] = 1.1  # Cauchy loss as default
        
        # Prepare initial parameter array
        init_params = np.array([initial_params[name] for name in self.param_names])
        
        # Run optimization
        options = {'maxiter': maxiter, 'disp': True}
        
        # #Different optimization methods may work better in different cases
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
        # if self.alpha_loss is None:
        #     bounds = [(0.001,0.5),(0.001,0.5),(0.6,0.999),(-5,5)]
        # else:
        #     bounds = [(0.001,0.5),(0.001,0.5),(0.6,0.999)]
        # result = differential_evolution(
        #     self._qle_objective,
        #     bounds= bounds,
        #     args=(y,),
        #     strategy='best1bin',
        #     maxiter=maxiter,
        #     disp=True,
        #     polish=True
        # )
        
        if not result.success:
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
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Forecast volatility for future steps.
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to forecast
        
        Returns:
        --------
        np.ndarray
            Forecasted volatility
        """
        if self.params is None:
            raise ValueError("Model must be fit before prediction")
        
        omega = self.params['omega']
        beta = self.params['beta']
        
        # For long-term forecasts, volatility approaches the unconditional variance
        forecast = np.zeros(steps)
        forecast[0] = self.fitted_volatility[-1]
        
        for t in range(1, steps):
            forecast[t] = omega + beta * forecast[t-1]
        
        return forecast
    
    def plot_volatility(self, y: np.ndarray, true_vol: np.ndarray, title: str = "Estimated Volatility") -> None:
        """
        Plot the estimated volatility with data.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed time series
        title : str
            Plot title
        """
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
        plt.plot(np.sqrt(true_vol), 'b-', alpha=0.3, label='True Volatility')
        plt.legend()
        plt.ylim(0, np.max(np.sqrt(self.fitted_volatility)) * 1.5)
        plt.tight_layout()
        plt.show()
    
    def simulate(self, T: int, dist: str = 't', df: int = 5, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate data from the model.
        
        Parameters:
        -----------
        T : int
            Number of observations to simulate
        dist : str
            Distribution of innovations ('t' for Student's t or 'norm' for normal)
        df : int
            Degrees of freedom for Student's t distribution
        seed : int
            Random seed
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Simulated returns and volatility
        """
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
            psi_t = self._rho_derivative(e_t, alpha_loss, self.c) * (-1)
            
            # Update volatility
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            f[t+1] = max(f[t+1], 1e-12)  # Ensure positive volatility
            
        return y, f[1:]

def grid_search_alpha(alpha_values):

        # Set parameters for the data generating process
    true_params = {
        'omega': 0.05,
        'gamma': 0.15,
        'beta': 0.8,
        'alpha_loss': 1,  # Cauchy loss
    }
    
    # Create a model with the true parameters for simulation
    sim_model = RobustQLEVolatilityModel(alpha_loss=true_params['alpha_loss'])
    sim_model.params = true_params
    
    # Simulate data
    T = 3000
    y, true_vol = sim_model.simulate(T, dist='t', df=5, seed=42)


    if alpha_values is not None:
        print("Warning: alpha_loss is already fixed, grid search not meaningful")
        model = RobustQLEVolatilityModel(alpha_loss=alpha_values)
        return model.fit(y)
    
    if alpha_values is None:
        alpha_values = np.concatenate([
            np.array([-2, -1, -0.5]),  # More robust than Cauchy
            np.linspace(0, 2, 11),     # Cauchy to L2 range (most common)
            np.array([3, 4])           # Less robust than L2
        ])

    best_alpha = None
    best_nll = float('inf')
    best_params = None

    for alpha in alpha_values:
        temp_model = RobustQLEVolatilityModel(alpha_loss=alpha)
        
        temp_params = temp_model.fit(y)
        nll = temp_model._qle_objective(temp_params, y)

        if nll < best_nll:
            best_nll = nll
            best_alpha = alpha
            best_params = temp_params
        print(f"Alpha: {alpha}, NLL: {nll:.4f}, Params: {temp_params}")
    return best_alpha, best_params    



# ----------------------
# Example usage
# ----------------------

def demo_simulated_data():
    """Demonstrate the model with simulated data"""
    # Set parameters for the data generating process
    true_params = {
        'omega': 0.07,
        'gamma': 0.11,
        'beta': 0.80,
        'alpha_loss': 1.1,  # Cauchy loss
    }
    
    # Create a model with the true parameters for simulation
    sim_model = RobustQLEVolatilityModel(alpha_loss=true_params['alpha_loss'])
    sim_model.params = true_params
    
    # Simulate data
    T = 4000
    y, true_vol = sim_model.simulate(T, dist='t', df=5, seed=40)
    
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


# True parameters:
#   omega: 0.0500
#   gamma: 0.1500
#   beta: 0.8000
#   alpha_loss: 1.0000

# Estimated parameters (fixed alpha):
#   omega: 0.2963
#   gamma: 0.4295
#   beta: 0.2622

# Estimated parameters (estimated alpha):
#   omega: 0.0255
#   gamma: 0.0223
#   beta: 0.9166
#   alpha_loss: 1.7005

# RMSE of volatility estimation (fixed alpha): 0.407746
# RMSE of volatility estimation (estimated alpha): 0.146929