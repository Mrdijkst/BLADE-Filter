# import numpy as np
# import tensorflow as tf
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt

# # -----------------------------------------------------------------------------
# # 1) Barron loss ρ(e; α,c) and score ψ = -∂ρ/∂e
# # -----------------------------------------------------------------------------
# def barron_rho(e, alpha, c):
#     z2 = (e / c)**2
#     a2 = tf.abs(alpha - 2.0)
#     is2 = tf.abs(alpha - 2.0) < 1e-6

#     rho2 = 0.5 * z2
#     rho0 = tf.math.log(0.5 * z2 + 1.0)
#     rho_gen = (a2 / alpha) * ((z2 / a2 + 1.0)**(alpha / 2.0) - 1.0)

#     return tf.where(is2, rho2, 
#            tf.where(tf.abs(alpha) < 1e-6, rho0, rho_gen))

# def compute_psi_and_jacobian(e_t, params):
#     """
#     Given e_t shape [T], and params = [alpha, beta, omega, gamma, c],
#     returns psi shape [T] and jacobian shape [T,5] = ∂psi/∂params.
#     """
#     alpha, beta, omega, gamma, c = params

#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(params)
#         with tf.GradientTape() as inner:
#             inner.watch(e_t)
#             rho = barron_rho(e_t, alpha, c)        # [T]
#         drho_de = inner.gradient(rho, e_t)         # [T]
#         psi = -drho_de                             # [T]

#     # Jacobian: list of [T] each ∂psi/∂param_i
#     jac_list = [tape.gradient(psi, p) for p in params]  # 5 × [T]
#     # stack to [T,5]
#     jacobian = tf.stack(jac_list, axis=1)
#     return psi, jacobian

# # -----------------------------------------------------------------------------
# # 2) Build G_T(θ) = (1/T) ∑ h_t/σ_t^2 ⋅ ∂f_t/∂θ via auto‑diff
# # -----------------------------------------------------------------------------
# def G_hat_tf(theta, data):
#     """
#     Returns G_hat shape [5] = (1/T) ∑ g_t where
#       g_t = (h_t / σ_t^2) * ∂f_t/∂θ   (vector of length 5)
#     """
#     T = tf.shape(data)[0]
#     alpha, beta, omega, gamma, c = theta

#     # initialize f_t
#     f = tf.TensorArray(tf.float32, size=T+1)
#     f = f.write(0, tf.math.reduce_variance(data))

#     # we will accumulate g_t across t
#     G_acc = tf.zeros((5,), dtype=tf.float32)

#     for t in tf.range(1, T):
#         f_t = f.read(t-1)

#         # compute e_t = data[t]^2 - f_t
#         e_t = data[t]**2 - f_t

#         # compute psi_t and jacobian_t = ∂ψ_t/∂θ
#         psi_t, jac_t = compute_psi_and_jacobian(
#             tf.reshape(e_t, (1,)),  # shape [1]
#             theta
#         )
#         psi_t = psi_t[0]       # scalar
#         jac_t = jac_t[0, :]    # [5]

#         # update f_t
#         f_new = omega + beta * f_t + gamma * psi_t
#         f = f.write(t, f_new)

#         # moment: h_t/σ_t^2 * ∂f_t/∂θ = (e_t)/(f_t) * jacobian of psi times gamma
#         # but ∂f_t/∂θ = [ ∂(ω + βf + γψ)/∂θ ] = [0, f_{t-1}, 0, ψ_t, 0] + γ ∂ψ/∂θ
#         df_dtheta = tf.stack([
#             tf.zeros([], dtype=tf.float32),      # ∂/∂α of ω-term
#             f_t,                                 # ∂/∂β (f_{t-1})
#             tf.ones([], dtype=tf.float32),       # ∂/∂ω
#             psi_t,                               # ∂/∂γ
#             tf.zeros([], dtype=tf.float32)       # ∂/∂c (only via psi)
#         ]) + gamma * jac_t                       # add γ ∂ψ/∂θ

#         h_t = e_t
#         G_acc += (h_t / f_t) * df_dtheta

#     # average
#     return G_acc / tf.cast(T, tf.float32)

# # -----------------------------------------------------------------------------
# # 3) Wrap objective and gradient for scipy
# # -----------------------------------------------------------------------------
# def objective_and_grad(theta_np, data_np):
#     # convert to tf
#     data = tf.constant(data_np, dtype=tf.float32)
#     theta = tf.constant(theta_np, dtype=tf.float32)

#     with tf.GradientTape() as tape:
#         tape.watch(theta)
#         G = G_hat_tf(theta, data)         # [5]
#         J = tf.reduce_sum(G * G)          # scalar
#     grad = tape.gradient(J, theta)       # [5]

#     return J.numpy().astype(float), grad.numpy().astype(float)

# # -----------------------------------------------------------------------------
# # 4) Estimation on simulated data
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # simulate GARCH‐like data
#     np.random.seed(42)
#     T = 500
#     omega_true, beta_true, gamma_true, alpha_true, c_true = 0.1, 0.8, 0.1, 1.0, 1.0
#     y = np.zeros(T)
#     f = np.zeros(T)
#     f[0] = omega_true / (1 - beta_true)
#     for t in range(1, T):
#         eps = np.random.randn()
#         y[t] = np.sqrt(f[t-1]) * eps
#         # psi = -∂ρ/∂e at true params
#         e = y[t]**2 - f[t-1]
#         # simple gaussian score
#         psi = -(e / c_true**2)
#         f[t] = omega_true + beta_true*f[t-1] + gamma_true*psi

#     # initial guess
#     theta0 = np.array([0.5, 0.5, 0.05, 0.05, 0.5], dtype=float)

#     res = minimize(
#         fun=lambda th: objective_and_grad(th, y)[0],
#         jac=lambda th: objective_and_grad(th, y)[1],
#         x0=theta0,
#         method="L-BFGS-B",
#         bounds=[(1e-6,None)]*5,
#         options={"disp": True}
#     )
#     print("Estimated θ:", res.x)

#     # 5) Plot fitted volatility vs proxy y^2
#     # re‐compute f_t with fitted parameters
#     theta_hat = res.x
#     fhat = np.zeros(T)
#     fhat[0] = np.var(y)
#     for t in range(1, T):
#         e = y[t]**2 - fhat[t-1]
#         # gaussian score for plotting
#         psi = -(e / theta_hat[4]**2)
#         fhat[t] = theta_hat[2] + theta_hat[1]*fhat[t-1] + theta_hat[3]*psi

#     plt.figure(figsize=(10,4))
#     plt.plot(y**2, label=r"$y_t^2$")
#     plt.plot(fhat, label=r"$\hat f_t$")
#     plt.legend()
#     plt.title("Empirical $y_t^2$ vs Fitted Volatility $\hat f_t$")
#     plt.show()

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class RobustQLEVolatilityModel:
    """
    Implementation of the Quasi-Likelihood Estimation (QLE) volatility model with
    robust loss function as proposed by Barron (2019) in the context of 
    Quasi-Score Driven models.
    """
    
    def __init__(self, alpha_loss: float = None, c: float = 1.0):
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
        if alpha == 2:
            # L2 loss (least squares)
            return e / (c**2)
        elif alpha == 0:
            # Cauchy/Lorentzian loss
            return (2 * e) / (e**2 + 2 * c**2)
        elif alpha == float('-inf'):
            # Welsch/Leclerc loss
            return (e / c**2) * np.exp(-0.5 * (e/c)**2)
        else:
            # General case
            return (e / c**2) * np.power((e**2 / c**2) / np.abs(alpha-2) + 1, alpha/2 - 1)
    
    #def _rho_second_derivative(self, e: np.ndarray, alpha: float, c: float) -> np.ndarray:
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
        if alpha == 2:
            # L2 loss (least squares)
            return np.ones_like(e) / (c**2)
        elif alpha == 0:
            # Cauchy/Lorentzian loss
            denom = (1 + 0.5 * (e/c)**2)**2
            return (1 / c**2) * (1 - 0.5 * (e/c)**2) / denom
        elif alpha == float('-inf'):
            # Welsch/Leclerc loss
            return (1 / c**2) * (1 - (e**2 / c**2)) * np.exp(-0.5 * (e/c)**2)
        else:
            # General case
            base_term = np.power((e/c)**2 / np.abs(alpha-2) + 1, alpha/2 - 2)
            part1 = (1 / c**2) * base_term
            part2 = (e/c)**2 / np.abs(alpha-2) + 1
            part3 = part2 + (alpha/2 - 1) * (2 * e**2) / (c**2 * np.abs(alpha-2))
            return part1 * part3
    
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
        f[0] = np.mean(y[:20]**2) if T >= 20 else np.mean(y**2)
        
        # Recursively update the volatility
        for t in range(T):
            e_t = y[t]**2 - f[t]
            psi_t = self._rho_derivative(e_t, alpha_loss, self.c) * (-1)  # -1 because e_t = y_t^2 - f_t
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            
            # Enforce positive volatility
            f[t+1] = max(f[t+1], 1e-6)
        
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
        
        # Initialize derivatives
        df_dtheta = np.zeros((T, n_params))
        
        for t in range(1, T):
            e_t = y[t-1]**2 - f[t-1]
            psi_t = self._rho_derivative(e_t, alpha_loss, self.c) * (-1)  # -1 because e_t = y_t^2 - f_t
            #d2psi_df = self._rho_second_derivative(e_t, alpha_loss, self.c)
            
            # Derivatives with respect to omega, gamma, beta
            df_dtheta[t, 0] = 1 + beta * df_dtheta[t-1, 0]
            df_dtheta[t, 1] = psi_t + beta * df_dtheta[t-1, 1]
            df_dtheta[t, 2] = f[t-1] + beta * df_dtheta[t-1, 2]
            
            # Derivative with respect to alpha_loss, if applicable
            if self.alpha_loss is None:
                # Partial derivative of psi with respect to alpha_loss
                # This is complex and depends on the specific form of the loss function
                # Numerical approximation could be used here
                delta = 1e-5
                psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) * (-1)
                psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) * (-1)
                dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
                
                df_dtheta[t, 3] = gamma * dpsi_dalpha + beta * df_dtheta[t-1, 3]
        
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
            
            # Compute residuals
            h_t = y**2 - f
            
            # Use f as the conditional variance approximation
            sigma2_t = f
            
            # Compute derivatives
            df_dtheta = self._compute_derivatives(y, f, params)
            
            # Form the QLE objective (Equation 3 in the thesis)
            G_t = np.sum(h_t.reshape(-1, 1) / sigma2_t.reshape(-1, 1) * df_dtheta, axis=0)
            obj = np.sum(G_t**2)
            
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
                'omega': 0.05, 
                'gamma': 0.15, 
                'beta': 0.8
            }
            if self.alpha_loss is None:
                initial_params['alpha_loss'] = 0.5  # Cauchy loss as default
        
        # Prepare initial parameter array
        init_params = np.array([initial_params[name] for name in self.param_names])
        
        # Run optimization
        options = {'maxiter': maxiter, 'disp': True}
        result = minimize(
            self._qle_objective, 
            init_params, 
            args=(y,), 
            method=method, 
            options=options
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
    
    def plot_volatility(self, y: np.ndarray, title: str = "Estimated Volatility") -> None:
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
        plt.plot(y**2, 'b-', alpha=0.3, label='Squared Returns')
        plt.legend()
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
            e_t = y[t]**2 - f[t]
            psi_t = self._rho_derivative(e_t, alpha_loss, self.c) * (-1)
            
            # Update volatility
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            f[t+1] = max(f[t+1], 1e-6)  # Ensure positive volatility
            
        return y, f[1:]


# ----------------------
# Example usage
# ----------------------

def demo_simulated_data():
    """Demonstrate the model with simulated data"""
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
    T = 10000
    y, true_vol = sim_model.simulate(T, dist='t', df=5, seed=42)
    
    # Create a model for estimation
    # Option 1: Fix alpha to a known value
    model1 = RobustQLEVolatilityModel(alpha_loss=0.0)  # Fix to Cauchy loss
    
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
    model1.plot_volatility(y, "QLE Volatility Model with Fixed Alpha")
    model2.plot_volatility(y, "QLE Volatility Model with Estimated Alpha")

    # Compare RMSE of volatility estimation
    rmse1 = np.sqrt(np.mean((model1.fitted_volatility - true_vol)**2))
    rmse2 = np.sqrt(np.mean((model2.fitted_volatility - true_vol)**2))
    
    print(f"\nRMSE of volatility estimation (fixed alpha): {rmse1:.6f}")
    print(f"RMSE of volatility estimation (estimated alpha): {rmse2:.6f}")
    
    return model1, model2, y, true_vol


def analyze_real_data(data: pd.Series):
    """Apply the model to real financial data"""
    # Calculate log returns
    returns = 100 * data.pct_change().dropna()
    
    # Create models with different loss functions
    model_l2 = RobustQLEVolatilityModel(alpha_loss=2.0)  # L2 loss
    model_cauchy = RobustQLEVolatilityModel(alpha_loss=0.0)  # Cauchy loss
    model_adaptive = RobustQLEVolatilityModel()  # Adaptive loss
    
    # Fit the models
    print("\nFitting L2 loss model:")
    params_l2 = model_l2.fit(returns.values, method='Nelder-Mead')
    
    print("\nFitting Cauchy loss model:")
    params_cauchy = model_cauchy.fit(returns.values, method='Nelder-Mead')
    
    print("\nFitting adaptive loss model:")
    params_adaptive = model_adaptive.fit(returns.values, method='Nelder-Mead')
    
    # Plot results
    model_l2.plot_volatility(returns.values, "QLE Volatility Model with L2 Loss")
    model_cauchy.plot_volatility(returns.values, "QLE Volatility Model with Cauchy Loss")
    model_adaptive.plot_volatility(returns.values, "QLE Volatility Model with Adaptive Loss")
    
    # Compare in-sample fit
    def compute_likelihood(y, vol, dist='t', df=5):
        """Compute log-likelihood assuming Student's t distribution"""
        from scipy.stats import t
        z = y / np.sqrt(vol)
        log_lik = np.sum(t.logpdf(z, df) - 0.5 * np.log(vol))
        return log_lik
    
    ll_l2 = compute_likelihood(returns.values, model_l2.fitted_volatility)
    ll_cauchy = compute_likelihood(returns.values, model_cauchy.fitted_volatility)
    ll_adaptive = compute_likelihood(returns.values, model_adaptive.fitted_volatility)
    
    print(f"\nLog-likelihood (L2 loss): {ll_l2:.2f}")
    print(f"Log-likelihood (Cauchy loss): {ll_cauchy:.2f}")
    print(f"Log-likelihood (Adaptive loss): {ll_adaptive:.2f}")
    
    return model_l2, model_cauchy, model_adaptive


if __name__ == "__main__":
    # Demo with simulated data
    model1, model2, sim_y, true_vol = demo_simulated_data()
    
    #Example with real data:
    # If you have data available, you can use it as follows:
    # from yahooquery import Ticker

    # fvx = Ticker("^AEX")
    # yahoo = fvx.history(start="2010-01-01", end="2024-01-01")
    # data = yahoo.iloc[:,5]
    
    # models = analyze_real_data(data)