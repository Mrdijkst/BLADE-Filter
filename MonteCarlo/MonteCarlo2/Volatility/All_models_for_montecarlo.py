import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.special import gammaln
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class RobustQLEModel:
    """
    Robust QLE  Model    
    """

    def __init__(self, model_type: str = 'volatility', alpha_loss: float = None, c: float = None):
        
        if model_type not in ['volatility', 'location']:
            raise ValueError("model_type must be either 'volatility' or 'location'")
        
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
        self.model_type = model_type
    
    def _rho_derivative(self, e: np.ndarray, alpha: float, c: float) -> np.ndarray:
        
        
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
    
        eps = 1e-4
    
        if abs(alpha_loss - 2) < eps:
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) 
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) 
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha  # ∂ψ/∂α = 0 for the L2 loss

        
        if abs(alpha_loss - 0) < eps:
            
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) 
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c) 
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha

        
        if alpha_loss < -1e3:
            delta = 1e-5
            psi_plus = self._rho_derivative(e_t, alpha_loss + delta, self.c) 
            psi_minus = self._rho_derivative(e_t, alpha_loss - delta, self.c)
            dpsi_dalpha = (psi_plus - psi_minus) / (2 * delta)
            return dpsi_dalpha  

      
        alpha= alpha_loss
        e2 = e_t**2
        denom = c**2 * np.abs(alpha - 2)
        A = (e2 / denom) + 1

        #
        term_log = np.log(A) / 2
        term_frac = (e2 * (alpha/2 - 1)) / (c**2 * A * np.abs(alpha - 2) * (alpha - 2))
        bracket = term_log - term_frac

      
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
            
    
    def _filter_volatility(
        self,
        y: np.ndarray,
        params: np.ndarray,
        f0: float | None = None
    ) -> np.ndarray:
            
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
            
            # Initialize state
            f = np.zeros(T + 1)

            if f0 is None:
                # In-sample filtering (original behavior)
                f[0] = omega / (1 - beta)
            else:
                # Out-of-sample continuation
                f[0] = f0
            
            # Recursively update the state
            for t in range(T):
                if self.model_type == 'volatility':
                    e_t = y[t]**2 - f[t]
                elif self.model_type == 'location':
                    e_t = y[t] - f[t]

                psi_t = self._rho_derivative(e_t, alpha_loss, c)   
                f[t+1] = omega + gamma * psi_t + beta * f[t]
                
                # Enforce positivity (volatility) / stability
                if self.model_type == 'volatility':
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
        if self.model_type == 'volatility':
            e0 = y[0]**2 - f[0]
        elif self.model_type == 'location':
            e0 = y[0] - f[0]
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
            if self.model_type == 'volatility':
                e_t = y[t-1]**2 - f[t-1]
            else:  # location
                e_t = y[t-1] - f[t-1]

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
            if self.model_type == 'volatility':
                if params[0] <= 0 or params[1] < 0 or params[2] < 0 or params[2] >= 1 or params[1] + params[2] >= 1: #or params[1] >= 0.3 or params[2] <= 0.5:
                    return 1e10
            else:  # location
                if abs(params[2]) >= 0.999:   # beta
                    return 1e10

            
            # Filter volatility
            f = self._filter_volatility(y, params)

            if self.model_type == 'volatility':
                e_t = y**2 - f
                h_t = self._rho_derivative(e_t, self.alpha_loss, self.c)
                sigma2_t = 2 * f**2
            else:  # location
                e_t = y - f
                h_t = self._rho_derivative(e_t, self.alpha_loss, self.c)
                sigma2_t = np.ones_like(f)
          
                        
           
            df_dtheta = self._compute_derivatives(y, f, params)
            
            
            G_t = np.sum(h_t.reshape(-1, 1) / sigma2_t.reshape(-1, 1) * df_dtheta, axis=0) / len(y)
            
            
            obj = np.linalg.norm(G_t)
            
            return obj
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e10
    
    def fit(self, y: np.ndarray, initial_params: Optional[Dict] = None, 
            method: str = 'Nelder-Mead', maxiter: int = 2000) -> Dict:
        
        
        if initial_params is None:
            if self.model_type == 'volatility':
                    initial_params = {
                        'omega': 0.02, 
                        'gamma': 0.11, 
                        'beta': 0.88
                    }
            else:  # location
                    initial_params = {
                        'omega': 0.0,
                        'gamma': 0.3, 
                        'beta': 0.5
                    }
                    
            if self.alpha_loss is None:
                    initial_params['alpha_loss'] = 0.85
            if self.c is None:
                    initial_params['c'] = 1.2
        
        # Prepare initial parameter array
        init_params = np.array([initial_params[name] for name in self.param_names])
        
        # Run optimization
        options = {'maxiter': maxiter, 'disp': False}
        
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
            
            if self.model_type == 'volatility':
                bounds.extend([(0.001, 0.5), (0.001, 0.5), (0.6, 0.999)])
            else:  # location
                bounds.extend([(-1.0, 1.0), (-1.0, 1.0), (-0.999, 0.999)])

            
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
                disp=False,
                polish=True
            )
        
        if not result.success and method != 'differential_evolution':
            print(f"Warning: Optimization did not converge: {result.message}")
            
            # Try again with different method if first one failed
            if method == 'Nelder-Mead':
                result = minimize(
                    self._qle_objective,
                    init_params,
                    args=(y,),
                    method='BFGS',
                    options={'maxiter': maxiter}
                )
        
        
        self.params = {name: val for name, val in zip(self.param_names, result.x)}
        
        # Compute fitted volatility
        param_array = np.array([self.params[name] for name in self.param_names])
        self.fitted_volatility = self._filter_volatility(y, param_array)
        if self.model_type == 'volatility':
            self.residuals = y**2 - self.fitted_volatility
        else:  # location
            self.residuals = y - self.fitted_volatility

        
        #print(f"Optimization result: {result.message}")
        #print(f"Parameters: {self.params}")
        
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
        plt.ylim(0, np.max(np.sqrt(true_vol)) * 1.5)
        plt.tight_layout()
        plt.show()
    
    def simulate(self, T: int, dist: str = 't', noise_scale: float = 1.0, df: int = 5, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        
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
        
       
        y = np.zeros(T)
        f = np.zeros(T+1)
        f[0] = omega / (1 - beta)  # Start at unconditional variance
        
        # Generate innovations
        if dist == 't':
            eps = np.random.standard_t(df, T)
        else:
            eps = np.random.normal(0, 1, T)
            
        for t in range(T):
            if self.model_type == 'volatility':
                y[t] = np.sqrt(f[t]) * eps[t]
                e_t = y[t]**2 - f[t]
            else:  # location
                y[t] = f[t] + eps[t] * noise_scale
                e_t = y[t] - f[t]

            psi_t = self._rho_derivative(e_t, alpha_loss, c)
            f[t+1] = omega + gamma * psi_t + beta * f[t]

            if self.model_type == 'volatility':
                f[t+1] = max(f[t+1], 1e-12)
            
        return y, f[1:]

    
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
        
        #result = minimize(
            objective,
            init_params,
            method='Nelder-mead',
            bounds=bounds,
            options={'disp': False, 'maxiter': 1000}
        #)

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





class GAS_Model:
    """
    GAS(1,1) equivalent to GARCH(1,1) under Normal assumption. This model is a volatility model.
    """
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
        options = {'maxiter': maxiter, 'disp': False}

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


class OracleStudentTLocation:
    """
    Oracle GAS location model with Student-t likelihood.
    Degrees of freedom nu is FIXED and equal to the DGP nu.
    """

    def __init__(self, y, nu):
        self.y = np.asarray(y, dtype=float)
        self.T = len(self.y)
        self.nu = nu

    # --------------------------------------------------
    # Negative log-likelihood + filtered states
    # --------------------------------------------------
    def _nll_and_states(self, params):
        omega, alpha, beta = params

        # basic constraints
        if alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10, None

        nu = self.nu
        y = self.y
        T = self.T

        theta = np.zeros(T)
        nll = 0.0

        # initialize at unconditional mean
        theta[0] = omega / (1.0 - beta)

        # Student-t constants
        const = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(np.pi * nu)
        )

        for t in range(T - 1):
            e = y[t] - theta[t]

            # log-density contribution
            logpdf = (
                const
                - 0.5 * np.log(nu)
                - ((nu + 1) / 2) * np.log(1 + (e * e) / nu)
            )
            nll -= logpdf

            # score for location
            score = (nu + 1) * e / (nu + e * e)

            # GAS update
            theta[t + 1] = omega + alpha * score + beta * theta[t]

        # last observation
        e_last = y[-1] - theta[-1]
        logpdf_last = (
            const
            - 0.5 * np.log(nu)
            - ((nu + 1) / 2) * np.log(1 + (e_last * e_last) / nu)
        )
        nll -= logpdf_last

        return nll, theta

    # --------------------------------------------------
    # Fit model
    # --------------------------------------------------
    def fit(self):
        init = np.array([0.0, 0.1, 0.8])  # omega, alpha, beta
        bounds = [
            (None, None),      # omega unrestricted
            (0.0, 0.999),      # alpha ≥ 0
            (0.0, 0.999)       # beta ≥ 0
        ]

        def obj(p):
            nll, _ = self._nll_and_states(p)
            return nll

        res = minimize(
            obj,
            init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000}
        )

        self.params = res.x
        _, self.theta_fitted = self._nll_and_states(self.params)
        return res

    # --------------------------------------------------
    # One-step-ahead predictions (PRE-UPDATE)
    # --------------------------------------------------
    def one_step_ahead(self, y_test):
        omega, alpha, beta = self.params
        nu = self.nu

        theta = self.theta_fitted[-1]
        preds = np.zeros(len(y_test))

        for t, yt in enumerate(y_test):
            preds[t] = theta
            e = yt - theta
            score = (nu + 1) * e / (nu + e * e)
            theta = omega + alpha * score + beta * theta

        return preds



## Not used in monte carlo
import numpy as np
from scipy.optimize import minimize

class StudentT_QLE_Location:
    """
    QLE estimation of a GAS location model using Student-t score.

    Model:
        y_t = f_t + eps_t
        f_{t+1} = omega + gamma * psi_t + beta * f_t

    with Student-t score:
        psi_t = (nu + 1) * e_t / (nu + e_t^2)
        e_t = y_t - f_t

    Estimation:
        Parameters (omega, gamma, beta) estimated by QLE
        nu is fixed (oracle / chosen)
    """

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def __init__(self, nu: float):
        self.nu = nu
        self.model_type = "location"
        self.param_names = ["omega", "gamma", "beta"]
        self.params = None
        self.fitted_state = None

    # --------------------------------------------------
    # Student-t score and derivative
    # --------------------------------------------------
    def _psi(self, e):
        """Student-t score for location."""
        nu = self.nu
        return (nu + 1.0) * e / (nu + e * e)

    def _dpsi_df(self, e):
        """
        Derivative of psi_t w.r.t. f_t.
        Since e = y - f, de/df = -1.
        """
        nu = self.nu
        dpsi_de = (nu + 1.0) * (nu - e * e) / (nu + e * e) ** 2
        return -dpsi_de

    # --------------------------------------------------
    # Filtering recursion
    # --------------------------------------------------
    def _filter(self, y, params):
        omega, gamma, beta = params
        T = len(y)

        f = np.zeros(T + 1)
        f[0] = omega / (1.0 - beta)

        for t in range(T):
            e_t = y[t] - f[t]
            psi_t = self._psi(e_t)
            f[t + 1] = omega + gamma * psi_t + beta * f[t]

        return f[1:]

    # --------------------------------------------------
    # Derivative recursion (Jacobian)
    # --------------------------------------------------
    def _compute_derivatives(self, y, f, params):
        omega, gamma, beta = params
        T = len(y)

        df_dtheta = np.zeros((T, 3))

        # t = 0
        e0 = y[0] - f[0]
        psi0 = self._psi(e0)
        dpsi0_df = self._dpsi_df(e0)

        df_dtheta[0, 0] = 1.0                 # d f1 / d omega
        df_dtheta[0, 1] = psi0                # d f1 / d gamma
        df_dtheta[0, 2] = f[0]                # d f1 / d beta

        # recursion
        for t in range(1, T):
            e = y[t - 1] - f[t - 1]
            psi = self._psi(e)
            dpsi_df = self._dpsi_df(e)

            common = gamma * dpsi_df + beta

            df_dtheta[t, 0] = 1.0 + common * df_dtheta[t - 1, 0]
            df_dtheta[t, 1] = psi + common * df_dtheta[t - 1, 1]
            df_dtheta[t, 2] = f[t - 1] + common * df_dtheta[t - 1, 2]

        return df_dtheta

    # --------------------------------------------------
    # QLE objective
    # --------------------------------------------------
    def _qle_objective(self, params, y):
        omega, gamma, beta = params

        # basic stability constraints
        if beta <= -0.999 or beta >= 0.999:
            return 1e10

        f = self._filter(y, params)
        e = y - f

        df_dtheta = self._compute_derivatives(y, f, params)

        # estimating equation
        G = np.mean(e[:, None] * df_dtheta, axis=0)

        return np.linalg.norm(G)

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(self, y):
        y = np.asarray(y)

        init = np.array([0.0, 0.2, 0.7])
        bounds = [
            (None, None),     # omega
            (None, None),     # gamma
            (-0.999, 0.999),  # beta
        ]

        res = minimize(
            self._qle_objective,
            init,
            args=(y,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000}
        )

        self.params = res.x
        self.fitted_state = self._filter(y, self.params)
        return res

    # --------------------------------------------------
    # One-step-ahead prediction (PRE-UPDATE)
    # --------------------------------------------------
    def one_step_ahead(self, y_test):
        omega, gamma, beta = self.params
        f = self.fitted_state[-1]

        preds = np.zeros(len(y_test))

        for t, yt in enumerate(y_test):
            preds[t] = f
            e = yt - f
            psi = self._psi(e)
            f = omega + gamma * psi + beta * f

        return preds
