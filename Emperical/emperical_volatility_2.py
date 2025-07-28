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
from  yahooquery import Ticker
from scipy.special import gammaln

class GAS_Model:
    def __init__(self, y):
        """
        GAS(1,1) equivalent to GARCH(1,1) under Normal assumption:
        link=f_identity (f_t = sigma2_t), scaling=identity
        Model: f_{t+1} = omega + alpha * y_t^2 + beta * f_t
        where f_t = sigma2_t.
        """
        self.y = np.asarray(y)
        self.T = len(y)

    def _update_f(self, params):
        omega, alpha, beta = params
        f = np.zeros(self.T)
        ll = 0.0
        # initialize f[0] at sample variance
        f[0] = np.var(self.y)
        for t in range(1, self.T):
            # conditional variance update (GAS=GARCH)
            f[t] = omega + alpha * self.y[t-1]**2 + beta * f[t-1]
            # log-likelihood contribution
            ll += -0.5 * (np.log(2*np.pi) + np.log(f[t]) + self.y[t]**2 / f[t])
        return -ll, f

    def fit(self, start=None):
        """Estimate GARCH-equivalent GAS parameters by ML"""
        maxiter = 1000
        verbose = True
         # Set default initial parameters if not provided
        
        initial_params = {
            'omega': 0.07, 
            'gamma': 0.11, 
            'beta': 0.8
        }
       
        
        # Prepare initial parameter array
        init_params = np.array([initial_params[name] for name in initial_params] )
        
        # Run optimization
        options = {'maxiter': maxiter, 'disp': verbose}

        bounds = [(1e-8, None), (0, None), (0, 1-1e-6)]

        options['adaptive'] = True  # Use adaptive Nelder-Mead for better convergence
        res = minimize(
            lambda params, y: self._update_f(params)[0],
            init_params, 
            args=(self.y,), 
            method= 'Nelder-Mead', 
            options=options,
        )
        self.params = res.x
        _, self.fitted_f = self._update_f(self.params)
        return res

    def get_fitted_variance(self):
        return self.fitted_f

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
    
    #def _rho_second_derivative(self, e: np.ndarray, alpha: float, c: float) -> np.ndarray:
        
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
    
    #def dpsi_dalpha(self, e_t, c, alpha_loss):
        
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

    
    #def dpsi_dc(self, e_t, c, alpha_loss):
       
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
        
        # Initialize volatility with a better estimate
        f = np.zeros(T+1)
        
        # Use empirical variance of returns as initial value
        # Take the mean of the first few squared returns to avoid impact of outliers
        initial_window = min(20, T)  # Use first 20 observations or less if T < 20
        empirical_var = np.mean(y[:initial_window]**2)
        
        # Combine empirical variance with unconditional variance
        # This provides a more robust starting point
        f[0] = 0.5 * (empirical_var + omega/(1-beta))
        
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
            if params[0] <= 0 or params[1] < 0 or params[2] < 0 or params[2] >= 1 or params[1]/params[4] + params[2] >= 0.9999999:
                return 1e10
            
            # Filter volatility
            f = self._filter_volatility(y, params)
            
            # Compute residuals - h_t is defined as y_t^k - f_t(θ)
            h_t = y**2 - f
            
            # Use f as the conditional variance approximation
            sigma2_t = f**2
            
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
                'omega': 0.10, 
                'gamma': 0.07, 
                'beta': 0.85
            }
            if self.alpha_loss is None:
                initial_params['alpha_loss'] = 0.8  # Default value close to Cauchy loss
            
            if self.c is None:
                initial_params['c'] = 1.2 # Default value for scale parameter
        
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
    
    
    
    
    
    def plot_volatility(self, y: np.ndarray, gas_vol= None,beta_t_vol= None, true_vol: np.ndarray = None, title: str = "Estimated Volatility") -> None:
        
        if self.fitted_volatility is None:
            raise ValueError("Model must be fit before plotting")
        
        plt.figure(figsize=(12, 8))
        # find place with outlier
        outlier_pos = np.where(np.abs(y) > 6)[0][0]
        print(outlier_pos)  
        # Plot original data
        plt.subplot(2, 1, 1)
        plt.plot(y[outlier_pos-100:outlier_pos+100], 'b-', alpha=0.5, label='Returns')
        plt.title(title)
        plt.legend()
        
        # Plot volatility
        plt.subplot(2, 1, 2)
        plt.plot(np.sqrt(self.fitted_volatility)[outlier_pos-100:outlier_pos+100], 'r-', label='Estimated Volatility')
        if gas_vol is not None:
            plt.plot(np.sqrt(gas_vol)[outlier_pos-100:outlier_pos+100], 'g-', label='GAS Volatility')
        plt.plot(np.abs(y)[outlier_pos-100:outlier_pos+100], alpha=0.3, label='Absolute Returns')
        if beta_t_vol is not None:
            plt.plot(np.sqrt(beta_t_vol)[outlier_pos-100:outlier_pos+100], 'y-', label='Beta-t Volatility')
        
        if true_vol is not None:
            plt.plot(np.sqrt(true_vol)[100:300], 'b-', alpha=0.3, label='True Volatility')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

        #general plot
        plt.figure(figsize=(12, 8))
        plt.plot(np.sqrt(self.fitted_volatility), 'r-', label='Estimated Volatility')
        plt.plot(np.sqrt(gas_vol), 'g-', label='GAS Volatility')
        plt.plot(np.sqrt(beta_t_vol), 'y-', label='Beta-t Volatility')
        plt.plot(np.abs(y), alpha=0.3, label='Absolute Returns')
        plt.legend()
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
            psi_t = self._rho_derivative(e_t, alpha_loss, c) 
            
            # Update volatility
            f[t+1] = omega + gamma * psi_t + beta * f[t]
            #f[t+1] = max(f[t+1], 1e-12)  # Ensure positive volatility
            
        return y, f[1:]

def data_from_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Parameters:
        ticker (str): Stock ticker symbol.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: DataFrame with historical stock prices.
    """
    stock = Ticker(ticker)
    data = stock.history(start=start, end=end)
    return data['close'].dropna()  # Return only the closing prices


def main():
    # Example usage
    ticker = 'GOOG'
    start_date = '2015-01-01'
    end_date = '2025-01-01'
    
    # Fetch data from Yahoo Finance
    #data = data_from_yahoo(ticker, start_date, end_date)
    # take first difference of the data
    #data = data.diff().dropna()

    #data.to_csv('data.csv')

    data = pd.read_csv('data.csv')['close']
    #standardize the data
    data = (data - data.mean()) / data.std()
    print(data.shape)

    pd.plot()
    
    

    # Compute empirical volatility
    
    # Fit the RobustQLEVolatilityModel
    

    model_gas = GAS_Model(data.values)
    model_gas.fit()
    # Get fitted volatility
    fitted_volatility = model_gas.get_fitted_variance()

    model_beta_t = Beta_t_GARCH11(data.values)
    model_beta_t.fit()
    fitted_volatility_beta_t = model_beta_t.get_fitted_variances()

    # do a warm start for the robust qle model
    initial_params ={}
    initial_params['omega'] = model_gas.params[0]
    initial_params['gamma'] = model_gas.params[1]
    initial_params['beta'] = model_gas.params[2]
    initial_params['alpha_loss'] =1
    initial_params['c'] = 1

    print(initial_params)
            
    model = RobustQLEVolatilityModel()
    model.fit(data.values, initial_params=initial_params)
    
    # Plot the results
    model.plot_volatility(gas_vol=fitted_volatility, beta_t_vol=fitted_volatility_beta_t, y=data.values)
if __name__ == "__main__":
    main()