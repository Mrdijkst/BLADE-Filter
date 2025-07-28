# import numpy as np
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt
# import tensorflow as tf


# def barron_loss(f, t, params, data):
#     alpha = params[0]
#     beta  = params[1]
#     omega = params[2]
#     gamma = params[3]
#     c= params[4]

#     f[t] = omega + beta * f[t-1] + gamma * s_f(f, t, data, params)  


# def s_f(f, t, params, data):
#     alpha = params[0]
#     c= params[4]
 
#     e = data**2 - barron_loss(f,t, params, data)

#     if alpha == 2: 
#         return (e / c ) * (-1)
#     elif alpha == 0:
#         return (2*e / (e**2+c**2))*(-1)
#     elif alpha == -np.inf:
#         return (2*e / (e^2+c^2))*(-1)
#     else:
#         return (e/c**2) * (e^2/(c**2 * np.abs(alpha-2)+1)**((alpha/2)-1)) * (-1)


# def functional_from(params, data):
#     alpha = params[0]
#     beta  = params[1]
#     omega = params[2]
#     gamma = params[3]
#     c= params[4]
#     T = len(data)

#     f = np.zeros(T, dtype=float)
#     f[0] = np.var(data)
#     sum = 0
#     for t in range(T):

#         f[t] = barron_loss(f, t, params, data)

#         h_t = data[t]^2 - f[t]

#         sigma_t = f[t]

#         with tf.GradientTape(persistent=True) as tape:
            
#             rho =  barron_loss(f, t, params, data)

#         derivative = tape.gradient(rho, [params])

#         derivative  = np.array(derivative)

#         sum += (h_t / sigma_t) * derivative
#     return (1/T) * sum
                    
# def minimize_functional(data, initial_params):
#     result = minimize(functional_from, initial_params, args=(data,), method='L-BFGS-B')
#     return result.x
        
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import differential_evolution
from autograd import grad, jacobian
warnings.filterwarnings('ignore')


class SimplifiedRobustQLEVolatilityModel:
    """
    Streamlined implementation of the Quasi-Likelihood Estimation (QLE) volatility model with
    robust loss function as proposed by Barron (2019).
    """
    
    def __init__(self, alpha_loss=None, c=1.0):
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
    
    def _robust_loss_derivatives(self, e, alpha, c):
        """
        Compute first and second derivatives of the robust loss function.
        
        Parameters:
        -----------
        e : error term (y_t^2 - f_t)
        alpha : robustness parameter
        c : scale parameter
        
        Returns:
        --------
        tuple: (first_derivative, second_derivative)
        """
        # Convert scalar to array if needed
        scalar_input = np.isscalar(e)
        if scalar_input:
            e = np.array([e])
        
        # First derivative calculation
        if alpha == 2:  # L2 loss
            first_deriv = e / (c**2)
        elif alpha == 0:  # Cauchy loss
            first_deriv = (2 * e) / (e**2 + 2 * c**2)
        elif alpha == float('-inf'):  # Welsch loss
            first_deriv = (e / c**2) * np.exp(-0.5 * (e/c)**2)
        else:  # General case
            first_deriv = (e / c**2) * np.power((e**2 / c**2) / np.abs(alpha-2) + 1, alpha/2 - 1)
        
        # Second derivative calculation
        if alpha == 2:  # L2 loss
            second_deriv = np.ones_like(e) / (c**2)
        elif alpha == 0:  # Cauchy loss
            denom = (1 + 0.5 * (e/c)**2)**2
            second_deriv = (1 / c**2) * (1 - 0.5 * (e/c)**2) / denom
        elif alpha == float('-inf'):  # Welsch loss
            second_deriv = (1 / c**2) * (1 - (e**2 / c**2)) * np.exp(-0.5 * (e/c)**2)
        else:  # General case
            base_term = np.power((e/c)**2 / np.abs(alpha-2) + 1, alpha/2 - 2)
            part1 = (1 / c**2) * base_term
            part2 = (e/c)**2 / np.abs(alpha-2) + 1
            part3 = part2 + (alpha/2 - 1) * (2 * e**2) / (c**2 * np.abs(alpha-2))
            second_deriv = part1 * part3
        
        # Return scalar if input was scalar
        if scalar_input:
            return first_deriv[0], second_deriv[0]
        return first_deriv, second_deriv
    
    def fit(self, y, initial_params=None, method='Nelder-Mead', maxiter=2000):
        """
        Fit the volatility model using QLE. This function handles parameter estimation,
        volatility filtering, and derivative computation in one step.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed time series
        initial_params : dict, optional
            Initial parameter values. If None, default values will be used.
        method : str
            Optimization method for scipy.optimize.minimize
        maxiter : int
            Maximum number of iterations for optimization
        
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        # Set default initial parameters if not provided
        if initial_params is None:
            initial_params = {
                'omega': 0.05, 
                'gamma': 0.15, 
                'beta': 0.8
            }
            if self.alpha_loss is None:
                initial_params['alpha_loss'] = 0.0  # Cauchy loss as default
        
        # Prepare initial parameter array
        init_params = np.array([initial_params[name] for name in self.param_names])
        
        # Define objective function with all operations combined
        def qle_objective(params, y):
            try:
                # Parameter constraints
                if self.alpha_loss is None and len(params) > 3:
                    if params[3] < -10 or params[3] > 10:
                        return 1e10
                
                # Basic parameter constraints
                if params[0] <= 0 or params[1] < 0 or params[2] < 0 or params[2] >= 1 or params[1] + params[2] >= 0.999:
                    return 1e10
                
                T = len(y)
                omega, gamma, beta = params[:3]
                
                if self.alpha_loss is None:
                    alpha_loss = params[3]
                else:
                    alpha_loss = self.alpha_loss
                
                # Initialize volatility and derivatives arrays
                f = np.zeros(T+1)
                f[0] = np.mean(y[:20]**2) if T >= 20 else np.mean(y**2)
                
                # Initialize derivatives of f_t with respect to theta
                n_params = 4 if self.alpha_loss is None else 3
                df_dtheta = np.zeros((T+1, n_params))
                
                # Combined filtering and derivative computation loop
                for t in range(T):
                    # Filter volatility
                    e_t = y[t]**2 - f[t]
                    psi_t_deriv, psi_t_second_deriv = self._robust_loss_derivatives(e_t, alpha_loss, self.c)
                    psi_t = psi_t_deriv * (-1)  # -1 because e_t = y_t^2 - f_t
                    
                    # Update volatility
                    f[t+1] = omega + gamma * psi_t + beta * f[t]
                    f[t+1] = max(f[t+1], 1e-6)  # Enforce positive volatility
                    
                    # Compute derivatives if t > 0 (because we need f[t-1])
                    if t > 0:
                        # Chain rule factor for all parameters
                        chain_factor = gamma * psi_t_second_deriv * (-1)**2 + beta
                        
                        # For omega (∂ω/∂θ_0 = 1, else 0)
                        df_dtheta[t+1, 0] = 1 + chain_factor * df_dtheta[t, 0]
                        
                        # For gamma (∂γ/∂θ_1 = 1, else 0)
                        df_dtheta[t+1, 1] = psi_t + chain_factor * df_dtheta[t, 1]
                        
                        # For beta (∂β/∂θ_2 = 1, else 0)
                        df_dtheta[t+1, 2] = f[t] + chain_factor * df_dtheta[t, 2]
                        
                        # For alpha_loss, if applicable
                        if self.alpha_loss is None:
                            # Approximate ∂ψ_t/∂α using numerical differentiation
                            delta = 1e-5
                            psi_plus, _ = self._robust_loss_derivatives(e_t, alpha_loss + delta, self.c)
                            psi_minus, _ = self._robust_loss_derivatives(e_t, alpha_loss - delta, self.c)
                            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta) * (-1)
                            
                            df_dtheta[t+1, 3] = gamma * dpsi_dalpha + chain_factor * df_dtheta[t, 3]
                
                # Extract the filtered volatility (skip the initialization)
                filtered_vol = f[1:]
                
                # Compute residuals - h_t = y_t^2 - f_t
                h_t = y**2 - filtered_vol
                
                # Extract derivatives (skip initialization row)
                df_dtheta = df_dtheta[1:, :]
                
                # Form the QLE estimating equation
                G_t = np.sum(h_t.reshape(-1, 1) / filtered_vol.reshape(-1, 1) * df_dtheta, axis=0) / T
                
                # The objective is to minimize ||G_t(θ)||²
                obj = np.sum(G_t**2)
                
                return obj
            except Exception as e:
                print(f"Error in objective function: {e}")
                return 1e10
        
        # Run optimization
        options = {'maxiter': maxiter, 'disp': True}
        if method == 'Nelder-Mead':
            options['adaptive'] = True
        
        result = minimize(
            qle_objective, 
            init_params, 
            args=(y,), 
            method=method, 
            options=options
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
            
            # Try again with different method if first one failed
            if method == 'Nelder-Mead':
                print("Trying BFGS method instead...")
                result = minimize(
                    qle_objective,
                    init_params,
                    args=(y,),
                    method='BFGS',
                    options={'maxiter': maxiter}
                )
        
        # Store parameters
        self.params = {name: val for name, val in zip(self.param_names, result.x)}
        
        # Compute fitted volatility (re-run the filter)
        T = len(y)
        omega = self.params['omega']
        gamma = self.params['gamma']
        beta = self.params['beta']
        alpha_loss = self.params.get('alpha_loss', self.alpha_loss)
        
        f = np.zeros(T+1)
        f[0] = np.mean(y[:20]**2) if T >= 20 else np.mean(y**2)
        
        for t in range(T):
            e_t = y[t]**2 - f[t]
            psi_t_deriv, _ = self._robust_loss_derivatives(e_t, alpha_loss, self.c)
            psi_t = psi_t_deriv * (-1)
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            f[t+1] = max(f[t+1], 1e-6)
        
        self.fitted_volatility = f[1:]
        self.residuals = y**2 - self.fitted_volatility
        
        print(f"Optimization result: {result.message}")
        print(f"Parameters: {self.params}")
        
        return self.params
    
    def analyze_and_predict(self, y, steps=1, plot=True, title="Estimated Volatility"):
        """
        Combined function for analysis, plotting, and prediction.
        
        Parameters:
        -----------
        y : np.ndarray
            Observed time series
        steps : int
            Number of steps ahead to forecast
        plot : bool
            Whether to plot the volatility
        title : str
            Plot title
        
        Returns:
        --------
        dict
            Dictionary with analysis results including:
            - forecasted_volatility: predicted volatility for future steps
            - rmse: in-sample root mean squared error
            - log_likelihood: in-sample log-likelihood assuming t distribution
        """
        if self.params is None:
            raise ValueError("Model must be fit before analysis")
        
        # Extract parameters
        omega = self.params['omega']
        beta = self.params['beta']
        
        # Forecast volatility
        forecast = np.zeros(steps)
        forecast[0] = self.fitted_volatility[-1]
        
        for t in range(1, steps):
            forecast[t] = omega + beta * forecast[t-1]
        
        # Compute RMSE
        rmse = np.sqrt(np.mean(self.residuals**2))
        
        # Compute log-likelihood assuming Student's t distribution with 5 df
        from scipy.stats import t
        df = 5  # Degrees of freedom
        z = y / np.sqrt(self.fitted_volatility)
        log_lik = np.sum(t.logpdf(z, df) - 0.5 * np.log(self.fitted_volatility))
        
        # Plot if requested
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Plot original data
            plt.subplot(2, 1, 1)
            plt.plot(y, 'b-', alpha=0.5, label='Returns')
            plt.title(title)
            plt.legend()
            
            # Plot volatility
            plt.subplot(2, 1, 2)
            plt.plot(np.sqrt(self.fitted_volatility), 'r-', label='Estimated Volatility')
            plt.plot(y**2, 'b-', alpha=0.3, label='Squared Returns')
            plt.legend()
            plt.ylim(0, np.max(np.sqrt(self.fitted_volatility)) * 1.5)
            plt.tight_layout()
            plt.show()
        
        # Return analysis results
        results = {
            'forecasted_volatility': forecast,
            'rmse': rmse,
            'log_likelihood': log_lik
        }
        
        return results
    
    def simulate(self, T, dist='t', df=5, seed=None):
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
        tuple
            Simulated returns and volatility
        """
        if self.params is None:
            raise ValueError("Parameters must be set before simulation")
        
        if seed is not None:
            np.random.seed(seed)
        
        omega = self.params['omega']
        gamma = self.params['gamma']
        beta = self.params['beta']
        alpha_loss = self.params.get('alpha_loss', self.alpha_loss)
        
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
            e_t = y[t]**2 - f[t]
            psi_t_deriv, _ = self._robust_loss_derivatives(e_t, alpha_loss, self.c)
            psi_t = psi_t_deriv * (-1)
            
            # Update volatility
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            f[t+1] = max(f[t+1], 1e-6)
            
        return y, f[1:]


# Example usage
def run_simplified_analysis():
    """Apply the simplified model to simulated and real data"""
    # Simulate data
    true_params = {
        'omega': 0.05,
        'gamma': 0.15,
        'beta': 0.8,
        'alpha_loss': 0.0,  # Cauchy loss
    }
    
    # Create model for simulation
    sim_model = SimplifiedRobustQLEVolatilityModel(alpha_loss=true_params['alpha_loss'])
    sim_model.params = true_params
    
    # Simulate data
    T = 2000
    y, true_vol = sim_model.simulate(T, dist='t', df=5, seed=42)
    
    # Create and fit model with estimated alpha
    model = SimplifiedRobustQLEVolatilityModel()
    params = model.fit(y, method='Nelder-Mead')
    
    # Analyze and predict
    results = model.analyze_and_predict(y, steps=10, title="Simplified QLE Volatility Model")
    
    # Print results
    print(f"\nRMSE: {results['rmse']:.6f}")
    print(f"Log-likelihood: {results['log_likelihood']:.2f}")
    print(f"Forecasted volatility (next 5 steps): {results['forecasted_volatility'][:5]}")
    
    return model, y, true_vol, results


# For real data analysis
def analyze_real_data(data):
    """Apply the simplified model to real financial data"""
    # Calculate log returns
    returns = 100 * data.pct_change().dropna()
    
    # Create and fit model with adaptive loss
    model = SimplifiedRobustQLEVolatilityModel()
    params = model.fit(returns.values, method='Nelder-Mead')
    
    # Analyze and predict
    results = model.analyze_and_predict(returns.values, steps=10, title="QLE Volatility Model - Real Data")
    
    print(f"\nParameters: {params}")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"Log-likelihood: {results['log_likelihood']:.2f}")
    print(f"Forecasted volatility (next 5 steps): {results['forecasted_volatility'][:5]}")
    
    return model, results


# Uncomment to run with real data
# from yahooquery import Ticker
# fvx = Ticker("^TNX")
# yahoo = fvx.history(start="2010-01-01", end="2024-01-01")
# data = yahoo.iloc[:,5]
# log_returns = pd.Series([np.log(data.iloc[i + 1] / data.iloc[i]) for i in range(len(data) - 1)])
# model, results = analyze_real_data(log_returns)

# Or run with simulated data
model, sim_y, true_vol, results = run_simplified_analysis()

    