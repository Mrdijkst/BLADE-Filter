import pandas as pd
import numpy as np
from Location_model_QLE3 import RobustQLELocationModel
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
# note this one uses e(y,f) = y_t - f_t for location model put use -\rho'(e)

from Gasmodel import GAS_Model

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
        f[0] = omega/(1-beta)
        
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
    #def _qle_objective(self, params: np.ndarray, y_input) -> float:
        # 1) Dwing y_input om naar numpy‐array
        if isinstance(y_input, pd.Series):
            y = y_input.values
        else:
            y = y_input
        
        # 2) Haal parameters (omega, gamma, beta, [alpha], [c]) uit params
        #    Zorg direct dat c > 1e-6:
        omega, gamma, beta = params[:3]
        idx = 3
        if self.alpha_loss is None:
            alpha_loss = params[idx]; idx += 1
        else:
            alpha_loss = self.alpha_loss
        if self.c is None:
            c = params[idx]
        else:
            c = self.c

        # 3) Striktere guard‐checks vóóraleer rekent te starten
        #    (a) c > 1e-6
        if c <= 1e-6:
            return 1e10
        #    (b) |alpha − 2| ≥ 1e-4, tenzij je de special‐case wilt forceren
        if self.alpha_loss is None:
            if abs(alpha_loss - 2.0) < 1e-4:
                # óf forceer tiny verplaatsing:
                alpha_loss = 2.0 + 1e-4
            if abs(alpha_loss - 0.0) < 1e-4:
                alpha_loss = 0.0 + 1e-4
        #    (c) Beta‐ en Gamma‐constraints:
        if beta < 0.5 or beta >= 1.0:
            return 1e10
        if gamma < 0.0:
            return 1e10
        if gamma + beta >= 0.999:
            return 1e10

        # 4) Bereken de gefilterde locatie f (NumPy‐array) met deze parameters
        try:
            f = self._filter_location(y, np.array([omega, gamma, beta] + 
                ([alpha_loss] if self.alpha_loss is None else []) +
                ([c]         if self.c is None else [])))
        except Exception:
            # Als filter al een fout geeft (bv delta overflow in psi), geef penalty
            return 1e10

        # 5) Reken de residuals en df/dtheta
        h_t = y - f
        sigma2_t = np.ones_like(f)  # (constante variantie in location‐model)

        # 6) Bereken afgeleiden df/dtheta; zorg dat die geen nan produceert
        try:
            df_dtheta = self._compute_derivatives(y, f, np.array([omega, gamma, beta] +
                                ([alpha_loss] if self.alpha_loss is None else []) +
                                ([c]         if self.c is None else [])))
        except Exception:
            return 1e10

        # 7) Test of er ergens in df_dtheta nan zit
        if np.isnan(df_dtheta).any():
            return 1e10

        # 8) Bereken G_t en objectief
        #    Zorg dat h_t en sigma2_t NumPy‐arrays zijn (dat hebben we al via y)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (h_t.reshape(-1,1) / sigma2_t.reshape(-1,1))
            # Als ratio of df_dtheta ooit NaN of Inf geeft, pennen we het ruimschoots
            if np.isnan(ratio).any():
                return 1e10
            G_t = np.sum(ratio * df_dtheta, axis=0) / len(y)
        # 9) Met normaals:
        if np.isnan(G_t).any():
            return 1e10

        obj = np.linalg.norm(G_t)
        if np.isnan(obj) or np.isinf(obj):
            return 1e10

        return obj

    def _qle_objective(self, params: np.ndarray, y: np.ndarray) -> float:
        
        try:
            # Apply parameter constraints
            param_idx = 3
            
            if self.alpha_loss is None:
                # Ensure alpha_loss is in reasonable range
                if params[param_idx] < -5 or params[param_idx] > 10:
                    return 1e10
                param_idx += 1
            
            if self.c is None:
                # Ensure c is positive and reasonable
                if params[param_idx] <= 0 or params[param_idx] > 2:
                    return 1e10
            
            # Basic parameter constraints for location model stability
            if  params[1] < 0 or params[2] < 0.5 or params[2] >= 1 or params[1] + params[2] >= 1:
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
        
        # Set default initial parameters if not provided
        if initial_params is None:
            # Use sample mean for omega initialization
            mean_y = np.mean(y)
            initial_params = {
                'omega': mean_y*0.2,  # Start with a fraction of the mean
                'gamma': 0.15, 
                'beta': 0.75
            }
            if self.alpha_loss is None:
                initial_params['alpha_loss'] = 2  # Default value close to Cauchy loss
            
            if self.c is None:
                initial_params['c'] = 0.5  # Use standard deviation for scale parameter
        
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
    #def fit(self, y: np.ndarray, initial_params: Optional[Dict] = None, 
        #method: str = 'Nelder-Mead', maxiter: int = 2000) -> Dict:
        """
        Pas hier de initialisatie aan door AR(1)-momentenschatter op y te gebruiken.
        """
        T = len(y)
        y_mean = np.mean(y)
        
        # 1) OLS-AR(1): bereken beta_hat en omega_hat
        #    (we centreren eerst op het gemiddelde voor stabiliteit):
        y_centered = y - y_mean
        num = np.sum(y_centered[1:] * y_centered[:-1])
        den = np.sum(y_centered[:-1]**2)
        beta_hat = num / den
        
        
        # 2) Omega bij stationariteit: omega_hat = y_mean * (1 - beta_hat)
        omega_hat = y_mean * (1 - beta_hat)
        
        # 3) Filter f^(0) met AR(1) om voorlopige residuen e^(0) te krijgen
        f0 = np.zeros(T)
        f0[0] = y_mean  # startwaarde
        for t in range(1, T):
            f0[t] = omega_hat + beta_hat * f0[t-1]
        e0 = y - f0
        
        # 4) Bereken residu-variantie en MAD(e0)
        sigma_u_hat = np.sqrt(np.sum(e0**2) / (T-1))   # L2-variantie van residuen
        # Voor c gebruiken we MAD:
        median_e0 = np.median(e0)
        mad_e0 = 1.4826 * np.median(np.abs(e0 - median_e0))
        
        # 5) Kies gamma_0 zodat Var(gamma * e0) ≈ sigma_u_hat^2
        #    In de praktijk is Var(e0) ≈ sigma_u_hat^2, dus gamma0 ≈ 1.
        gamma_hat = 0.8  # een iets conservatievere start (tussen 0.5 en 1.0)
        
        # 6) Kies alpha_0 en c_0 (indien zelf te optimaliseren)
        alpha0 = 0.5     # start in Cauchy-gebied
        c0 = 1

        
        # 7) Vul initial_params aan (als gebruiker niets doorgeeft)
        if initial_params is None:
            initial_params = {
                'omega': omega_hat,
                'gamma': gamma_hat,
                'beta': beta_hat
            }
            if self.alpha_loss is None:
                initial_params['alpha_loss'] = alpha0
            if self.c is None:
                initial_params['c'] = c0
        
        # 8) Zet de initial_params om in array volgorde volgens self.param_names
        init_params = np.array([initial_params[name] for name in self.param_names])

        print(">>> init_params =", init_params)
        print(">>> objective(init_params) =", self._qle_objective(init_params, y))


        
        # 9) Start de optimalisatie (Nelder-Mead, BFGS, etc.) zoals eerder.
        #    ...
        options = {'maxiter': maxiter, 'disp': True}
        if method == 'Nelder-Mead':
            options['adaptive'] = True
            result = minimize(
                self._qle_objective,
                init_params,
                args=(y,),
                method='Nelder-Mead',
                options=options
            )
        elif method == 'BFGS':
            opts = {'maxiter': maxiter, 'gtol': 1e-6}
            result = minimize(
                self._qle_objective,
                init_params,
                args=(y,),
                method='BFGS',
                options=opts
            )
        elif method == 'differential_evolution':
            # ... voer bounds in ...
            bounds = [(-2, 2), (0.001, 0.5), (0.6, 0.999)]
            if self.alpha_loss is None:
                bounds.append((-5, 5))
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
        # 10) Indien niet-convergerend, probeer alternatieve methode
        if not result.success and method != 'differential_evolution':
            print(f"Waarschuwing: Optimizer converteerde niet: {result.message}")
            if method == 'Nelder-Mead':
                print("Probeer BFGS in plaats van Nelder-Mead …")
                result = minimize(
                    self._qle_objective,
                    init_params,
                    args=(y,),
                    method='BFGS',
                    options={'maxiter': maxiter}
                )
        
        # 11) Sla resultaat op en bereken fitted_location, residuals, …
        self.params = {name: val for name, val in zip(self.param_names, result.x)}
        param_array = np.array([self.params[name] for name in self.param_names])
        self.fitted_location = self._filter_location(y, param_array)
        self.residuals = y - self.fitted_location
        
        print(f"Optimalisatie resultaat: {result.message}")
        print(f"Beginwaarden gebruikt: omega={omega_hat:.4f}, beta={beta_hat:.4f}, "
            f"gamma={gamma_hat:.4f}, alpha={initial_params.get('alpha_loss', None)}, c={initial_params.get('c', None)}")
        print(f"Gevonden parameters: {self.params}")
        return self.params
    def plot_location(self, gas_est: None, date_index : None , y: np.ndarray, true_loc: np.ndarray = None, title: str = "Estimated Location") -> None:
        
        if self.fitted_location is None:
            raise ValueError("Model must be fit before plotting")
        
        plt.figure(figsize=(12,6))
        # Plot de fitted location tegen dezelfde index
        plt.plot(date_index, self.fitted_location, 'r-', linewidth=2, label='Estimated Location')
        # Plot de data zelf (y) tegen dezelfde index
        plt.plot(date_index, y, 'g--', alpha=0.8, label='Transformed Data')
        plt.plot(date_index, gas_est, 'b--', alpha=0.8, label='Gas Estimation')
        plt.title(title)
        plt.ylim(50,80)
        plt.legend()
        plt.tight_layout()
        plt.show()



class GAS_Location_Model:
    def __init__(self, y):
        """
        GAS(1,1) location model:
        y_t = mu_t + e_t,   e_t ~ N(0, sigma^2)
        link=f_identity (f_t = mu_t), scaling=identity
        Model: f_{t+1} = omega + alpha * s_t + beta * f_t
        where s_t = score of the log-likelihood w.r.t. mu_t
        """
        self.y = np.asarray(y)
        self.T = len(y)

    def _update_f(self, params):
        omega, alpha, beta = params
        mu = np.zeros(self.T)
        ll = 0.0
        mu[0] = np.mean(self.y)  # initialize
        sigma2 = np.var(self.y)  # fixed variance for simplicity

        for t in range(1, self.T):
            score_t = (self.y[t-1] - mu[t-1]) / sigma2  # dlogL/dmu_{t-1}
            mu[t] = omega + alpha * score_t + beta * mu[t-1]
            ll += -0.5 * (np.log(2 * np.pi * sigma2) + (self.y[t] - mu[t])**2 / sigma2)

        return -ll, mu

    def fit(self):
        """Estimate GAS location model parameters by ML"""
        maxiter = 1000
        verbose = True

        initial_params = {
            'omega': 0.0,
            'alpha': 0.1,
            'beta': 0.8
        }

        init_params = np.array([initial_params[name] for name in initial_params])

        options = {'maxiter': maxiter, 'disp': verbose, 'adaptive': True}
        res = minimize(
            lambda params, y: self._update_f(params)[0],
            init_params,
            args=(self.y,),
            method='Nelder-Mead',
            options=options
        )
        self.params = res.x
        _, self.fitted_mu = self._update_f(self.params)
        return res

    def get_fitted_location(self):
        return self.fitted_mu
    

def load_and_preprocess_data(file_path, price_area='DK2'):
    """
    Load and preprocess electricity spot price data by splitting each day into 
    four quartiles and calculating the mean price for each quartile.
    
    This approach provides 4 observations per day instead of weekly Monday averages,
    capturing intraday price dynamics while maintaining temporal aggregation.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing Nord Pool spot prices
    price_area : str
        Price area to filter for (e.g., 'DK1', 'DK2')
    
    Returns:
    --------
    y : numpy.ndarray
        Quartile time-series constructed by taking the mean electricity spot price 
        for each quartile of the day, taking logs and multiplying by 10
    quartile_prices : pandas.Series
        Quartile mean spot prices time series (for analysis)
    """
    import pandas as pd
    import numpy as np
    
    print(f"Loading Nord Pool electricity spot price data from: {file_path}")
    print("Methodology: Daily quartile time-series by taking mean spot price for each 6-hour period")
    
    # Load the dataset with proper separator
    try:
        df = pd.read_csv(file_path, sep=';')
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Date range: {df['HourDK'].min()} to {df['HourDK'].max()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    # Fix datetime parsing
    df['HourDK'] = pd.to_datetime(df['HourDK'], errors='coerce')
    
    # Fix numeric columns - handle European decimal format (comma as decimal separator)
    print("Converting price columns to numeric...")
    
    def convert_european_decimal(series):
        """Convert European decimal format (comma as decimal separator) to float"""
        if series.dtype == 'object':
            # Replace comma with dot for decimal separator
            return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')
        else:
            return series
    
    # Convert price columns
    df['SpotPriceDKK'] = convert_european_decimal(df['SpotPriceDKK'])
    df['SpotPriceEUR'] = convert_european_decimal(df['SpotPriceEUR'])
    
    # Check for conversion issues
    dkk_na = df['SpotPriceDKK'].isna().sum()
    eur_na = df['SpotPriceEUR'].isna().sum()
    
    print(f"Price conversion complete. DKK NAs: {dkk_na}, EUR NAs: {eur_na}")
    
    if dkk_na > 0:
        print(f"Warning: {dkk_na} DKK prices could not be converted")
    if eur_na > 0:
        print(f"Warning: {eur_na} EUR prices could not be converted")
    
    # Check for parsing errors
    if df['HourDK'].isna().any():
        print(f"Warning: {df['HourDK'].isna().sum()} datetime parsing errors found")
        df = df.dropna(subset=['HourDK'])
    
    # Remove rows with invalid price data
    df = df.dropna(subset=['SpotPriceDKK'])
    
    # Filter for specific price area
    if price_area:
        available_areas = df['PriceArea'].unique()
        print(f"Available price areas: {available_areas}")
        
        if price_area not in available_areas:
            print(f"Warning: {price_area} not found. Using first available area: {available_areas[0]}")
            price_area = available_areas[0]
        
        df = df[df['PriceArea'] == price_area]
        print(f"Filtered for {price_area}. Remaining rows: {len(df)}")
    
    # Set datetime as index and filter date range
    df = df.set_index('HourDK')
    df = df[df.index > '2016-09-01']
    df = df[df.index < '2021-09-01']
    print(f"Date filtered data. Remaining rows: {len(df)}")
    
    # Create date and hour columns for quartile calculation
    df['Date'] = df.index.date
    df['Hour'] = df.index.hour
    
    # Define quartiles based on hour of day (0-23)
    # Quartile 1: 00:00-05:59 (hours 0-5)    - Night/Early Morning
    # Quartile 2: 06:00-11:59 (hours 6-11)   - Morning
    # Quartile 3: 12:00-17:59 (hours 12-17)  - Afternoon
    # Quartile 4: 18:00-23:59 (hours 18-23)  - Evening/Night
    
    def assign_quartile(hour):
        if 0 <= hour <= 5:
            return 1
        elif 6 <= hour <= 11:
            return 2
        elif 12 <= hour <= 17:
            return 3
        else:  # 18 <= hour <= 23
            return 4
    
    df['Quartile'] = df['Hour'].apply(assign_quartile)
    
    print("Daily quartile assignment:")
    print("Quartile 1: 00:00-05:59 (Night/Early Morning)")
    print("Quartile 2: 06:00-11:59 (Morning)")
    print("Quartile 3: 12:00-17:59 (Afternoon)")
    print("Quartile 4: 18:00-23:59 (Evening/Night)")
    
    # Calculate quartile means for each day
    quartile_data = df.groupby(['Date', 'Quartile'])['SpotPriceDKK'].mean().reset_index()
    
    # Create a proper datetime index for each quartile
    # We'll use the middle hour of each quartile as representative time
    quartile_hours = {1: 3, 2: 9, 3: 15, 4: 21}  # Representative hours for each quartile
    
    quartile_data['DateTime'] = pd.to_datetime(quartile_data['Date']) + \
                               pd.to_timedelta(quartile_data['Quartile'].map(quartile_hours), unit='h')
    
    # Set datetime as index and sort
    quartile_data = quartile_data.set_index('DateTime').sort_index()
    
    # Extract the price series
    quartile_prices = quartile_data['SpotPriceDKK']
    
    print(f"Quartile observations: {len(quartile_prices)} (should be ~4x daily observations)")
    print(f"Quartile price statistics - Min: {quartile_prices.min():.2f} DKK, Max: {quartile_prices.max():.2f} DKK, Mean: {quartile_prices.mean():.2f} DKK")
    
    # Check quartile distribution
    quartile_counts = quartile_data['Quartile'].value_counts().sort_index()
    print("Quartile distribution:")
    for q in [1, 2, 3, 4]:
        count = quartile_counts.get(q, 0)
        print(f"  Quartile {q}: {count} observations")
    
    # Remove any remaining NaN values
    quartile_prices = quartile_prices.dropna()
    print(f"After removing NaN: {len(quartile_prices)} quartile observations")
    
    if len(quartile_prices) == 0:
        print("Error: No valid quartile prices after preprocessing")
        return None, None
    
    # Handle negative or zero prices (if any)
    if (quartile_prices <= 0).any():
        print("Warning: Non-positive prices found. Adding small constant before log transform")
        quartile_prices = quartile_prices + abs(quartile_prices.min()) + 1
    
    # Apply transformation: take logs and multiply by 10
    log_prices = np.log(quartile_prices) * 10
    y_log = np.log(quartile_prices.values) * 10

    
    # Final time series
    y = y_log
    
    print(f"\nTransformation complete with quartile methodology:")
    print(f"- Daily time-series constructed by splitting each day into 4 quartiles (6-hour periods)")
    print(f"- Taking mean electricity spot price for each quartile")
    print(f"- Taking logs of quartile means and multiplying by 10")
    print(f"- Centering around mean for stability")
    print(f"Final array shape: {y.shape}")
    print(f"Transformed price statistics - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")
    
    # Add quartile information to the series for later analysis
    quartile_prices_with_info = quartile_data[['SpotPriceDKK', 'Quartile']].copy()
    quartile_prices_with_info['LogTransformed'] = y
    
    return y, quartile_prices


def analyze_quartile_patterns(quartile_data):
    """
    Analyze price patterns across different quartiles of the day
    
    Parameters:
    -----------
    quartile_data : pandas.DataFrame
        DataFrame containing quartile price data
    
    Returns:
    --------
    analysis : dict
        Dictionary containing quartile analysis results
    """
    import pandas as pd
    import numpy as np
    
    # Group by quartile to analyze patterns
    quartile_stats = quartile_data.groupby('Quartile')['SpotPriceDKK'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(2)
    
    print("\nQuartile Price Analysis:")
    print("=" * 60)
    print("Quartile 1 (00:00-05:59): Night/Early Morning")
    print("Quartile 2 (06:00-11:59): Morning") 
    print("Quartile 3 (12:00-17:59): Afternoon")
    print("Quartile 4 (18:00-23:59): Evening/Night")
    print("-" * 60)
    print(quartile_stats)
    
    # Calculate relative differences
    overall_mean = quartile_data['SpotPriceDKK'].mean()
    
    print(f"\nRelative to Overall Mean ({overall_mean:.2f} DKK):")
    print("-" * 40)
    for q in [1, 2, 3, 4]:
        q_mean = quartile_stats.loc[q, 'mean']
        diff_pct = ((q_mean - overall_mean) / overall_mean) * 100
        print(f"Quartile {q}: {diff_pct:+.1f}% ({q_mean:.2f} DKK)")
    
    return quartile_stats


def main_with_quartiles():
    """
    Main execution function using quartile-based preprocessing
    """
    # File path
    file_path = "/Users/MathijsDijkstra/Downloads/Elspotprices.csv"
    
    print("="*80)
    print("EMPIRICAL ILLUSTRATION: Nord Pool Electricity Spot Prices")
    print("QUARTILE-BASED ANALYSIS")
    print("="*80)
    print("Analyzing quartile electricity spot prices - 4 observations per day")
    print("Each day split into 6-hour periods for enhanced temporal resolution")
    print("-"*80)
    
    # Load and preprocess data with quartile approach
    y, quartile_prices = load_and_preprocess_data(file_path, price_area='DK2')
    
    if y is None:
        print("Failed to load and preprocess data. Exiting.")
        return None, None, None
    
    # Additional quartile analysis if we have the detailed data
    try:
        # Recreate quartile analysis data for pattern analysis
        print("\nPerforming quartile pattern analysis...")
        # This would require the detailed quartile_data from the function
        # For now, we'll show basic statistics
        
        print(f"\nBasic Time Series Statistics:")
        print(f"Total observations: {len(y)}")
        print(f"Expected daily observations: ~4 (one per quartile)")
        print(f"Approximate days covered: {len(y) // 4}")
        
    except Exception as e:
        print(f"Quartile analysis error: {e}")
    
    print("-"*80)
    print("QUARTILE DATA PREPROCESSING COMPLETE")
    print("-"*80)
    print("Ready for model fitting with enhanced temporal resolution")
    print(f"Time series length: {len(y)} quartile observations")
    
    return y, quartile_prices, None

def fit_location_model(y, alpha_loss=None, **kwargs):
    """
    Fit the RobustQLELocationModel to the data
    
    Parameters:
    -----------
    y : numpy.ndarray
        Preprocessed price data
    alpha_loss : float or None
        Alpha parameter for the loss function
    **kwargs : dict
        Additional parameters for the model
    
    Returns:
    --------
    model : RobustQLELocationModel
        Fitted model instance
    """
    
    print(f"Initializing RobustQLELocationModel with alpha_loss={alpha_loss}")
    
    try:
        # Initialize the model
        model = RobustQLELocationModel( **kwargs)
        
        # Fit the model
        print("Fitting model to data...")
        model.fit(y)
        
        print("Model fitting completed successfully")
        return model
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    y, monday_prices, _ = main_with_quartiles()

    model = fit_location_model(y)
    model_gas =  GAS_Location_Model(y)
    model_gas.fit()
    if model is None:
        print("Failed to fit model. Exiting.")
    
    # Plot results

    print("-"*80)
    print("GENERATING RESULTS AND PLOTS")
    print("-"*80)
    
    print("Generating main model plot...")
    model.plot_location( model_gas.get_fitted_location(), monday_prices.index,  y, title="Estimated Weekly Location - Nord Pool Spot Price (Denmark DK2)")
    
    # Force display and keep window open
    plt.show(block=True)
    
    # Alternative: Save the plot
    # plt.savefig('nord_pool_location_model.png', dpi=300, bbox_inches='tight')
    # print("Main plot saved as 'nord_pool_location_model.png'")
    
    # Additional analysis plots showing the data transformation process
    print("Generating data analysis plots...")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Original Monday average spot prices
    plt.subplot(2, 3, 1)
    monday_prices.plot(color='blue', alpha=0.7)
    plt.title("Weekly Monday Average Spot Prices\n(Nord Pool DK2)")
    plt.ylabel("Price (DKK/MWh)")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Log-transformed prices (final time series used in model)
    plt.subplot(2, 3, 2)
    plt.plot(monday_prices.index, y, color='red', alpha=0.8)
    plt.title("Log-Transformed Weekly Time Series\n(log(Monday_avg) × 10)")
    plt.ylabel("Log Price × 10")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of original Monday prices
    plt.subplot(2, 3, 3)
    plt.hist(monday_prices, bins=50, alpha=0.7, edgecolor='black', color='blue')
    plt.title("Distribution of Monday Prices")
    plt.xlabel("Price (DKK/MWh)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of log-transformed prices
    plt.subplot(2, 3, 4)
    plt.hist(y, bins=50, alpha=0.7, edgecolor='black', color='red')
    plt.title("Distribution of Log-Transformed Prices")
    plt.xlabel("Log Price × 10")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Time series comparison (original vs transformed)
    plt.subplot(2, 3, 5)
    # Normalize both series for comparison
    norm_monday = (monday_prices - monday_prices.mean()) / monday_prices.std()
    norm_y = (y - y.mean()) / y.std()
    plt.plot(monday_prices.index, norm_monday, label='Original (normalized)', alpha=0.7)
    plt.plot(monday_prices.index, norm_y, label='Log-transformed (normalized)', alpha=0.7)
    plt.title("Original vs Log-Transformed Series\n(Both Normalized)")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
