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
                if params[param_idx] <= 0 or params[param_idx] > 2:
                    return 1e10
            
            # Basic parameter constraints for location model stability
            if params[0] <= -10 or params[0] >= 10 or params[1] < 0 or params[2] < 0.5 or params[2] >= 1 or params[1] + params[2] >= 0.999:
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
                'omega': 0.05,  # Start with a fraction of the mean
                'gamma': 0.2, 
                'beta': 0.75
            }
            if self.alpha_loss is None:
                initial_params['alpha_loss'] = 1  # Default value close to Cauchy loss
            
            if self.c is None:
                initial_params['c'] = 0.8  # Use standard deviation for scale parameter
        
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
        
        if y is not None:
            plt.plot((y-y.mean())/np.std(y) , 'g--', alpha=0.8, label='Transformed Data')
        
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

def load_and_preprocess_data(file_path, price_area='DK2'):
    """
    Load and preprocess electricity spot price data following the methodology
    of Escribano et al. (2011) for electricity price analysis.
    
    We analyze weekly electricity spot prices from the Nord Pool market (Denmark).
    Electricity prices are characterized by random walk dynamics contaminated by 
    unexpected occurrences of much higher prices during short periods. This erratic 
    behaviour is primarily caused by the non-storability of electricity, which means 
    that demand and supply must always be balanced, implying that shocks in either 
    supply or demand will induce large price movements.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing Nord Pool spot prices
    price_area : str
        Price area to filter for (e.g., 'DK1', 'DK2')
    
    Returns:
    --------
    y : numpy.ndarray
        Weekly time-series constructed by taking the average electricity spot price 
        on Monday, taking logs of this average and multiplying it by 10
    monday_prices : pandas.Series
        Average Monday spot prices time series (for analysis)
    """
    
    print(f"Loading Nord Pool electricity spot price data from: {file_path}")
    print("Following methodology: weekly time-series by taking average spot price on Monday")
    
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
    
    # Set datetime as index
    df = df.set_index('HourDK')
    df = df[df.index < '2021-09-01']
    
    # Extract day of week (Monday = 0)
    df['DayOfWeek'] = df.index.dayofweek
    df['Date'] = df.index.date
    
    # Filter for Monday prices only (following the methodology)
    monday_df = df[df['DayOfWeek'] == 0].copy()
    print(f"Monday observations: {len(monday_df)}")
    
    if len(monday_df) == 0:
        print("Error: No Monday observations found")
        return None, None
    
    # Calculate daily averages for Mondays (in case multiple hourly observations per Monday)
    monday_prices = df.groupby('Date')['SpotPriceDKK'].mean()
    monday_prices.index = pd.to_datetime(monday_prices.index)
    
    print(f"Weekly Monday averages: {len(monday_prices)} observations")
    print(f"Monday price statistics - Min: {monday_prices.min():.2f} DKK, Max: {monday_prices.max():.2f} DKK, Mean: {monday_prices.mean():.2f} DKK")
    
    # Remove any remaining NaN values
    monday_prices = monday_prices.dropna()
    print(f"After removing NaN: {len(monday_prices)} Monday observations")
    
    if len(monday_prices) == 0:
        print("Error: No valid Monday prices after preprocessing")
        return None, None
    
    # Handle negative or zero prices (if any)
    if (monday_prices <= 0).any():
        print("Warning: Non-positive prices found. Adding small constant before log transform")
        monday_prices = monday_prices + abs(monday_prices.min()) + 1
    
    # Apply transformation: take logs and multiply by 10 (following methodology)
    log_prices = np.log(monday_prices) * 10
    
    # Final check and convert to numpy array
    y = monday_prices.values

    y = (y- y.mean()) / y.std()  # Normalize the series for better model fitting
    
    print(f"\nTransformation complete following methodology:")
    print(f"- Weekly time-series constructed by taking average electricity spot price on Monday")
    print(f"- Taking logs of this average and multiplying by 10")
    print(f"Final array shape: {y.shape}")
    print(f"Transformed price statistics - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")
    
    return y, monday_prices

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

def main():
    """
    Main execution function
    
    Empirical illustration analyzing weekly electricity spot prices from Nord Pool (Denmark).
    Time-series of electricity prices are characterized by random walk dynamics contaminated 
    by unexpected occurrences of much higher prices during short periods. This erratic 
    behaviour is primarily caused by the non-storability of electricity, which means that 
    demand and supply must always be balanced, implying that shocks in either supply or 
    demand will induce large price movements.
    
    We construct a weekly time-series by taking the average electricity spot price on 
    Monday, taking logs of this average and multiplying it by 10.
    """
    
    # File path
    file_path = "/Users/MathijsDijkstra/Downloads/Elspotprices.csv"
    
    print("="*80)
    print("EMPIRICAL ILLUSTRATION: Nord Pool Electricity Spot Prices")
    print("="*80)
    print("Analyzing weekly electricity spot prices following methodology of")
    print("Escribano et al. (2011) for electricity price dynamics analysis")
    print("-"*80)
    
    # Load and preprocess data
    y, monday_prices = load_and_preprocess_data(file_path, price_area='DK2')
    
    if y is None:
        print("Failed to load and preprocess data. Exiting.")
        return
    
    print("-"*80)
    print("FITTING ROBUST QLE LOCATION MODEL")
    print("-"*80)
    
    # Fit the model
    model = fit_location_model(y, alpha_loss=None)
    
    if model is None:
        print("Failed to fit model. Exiting.")
        return
    
    # Plot results
    try:
        print("-"*80)
        print("GENERATING RESULTS AND PLOTS")
        print("-"*80)
        
        print("Generating main model plot...")
        model.plot_location(y, title="Estimated Weekly Location - Nord Pool Spot Price (Denmark DK2)")
        
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
        
        # Plot 6: Descriptive statistics text
        plt.subplot(2, 3, 6)
        plt.axis('off')
        stats_text = f"""
DATA SUMMARY

Original Monday Prices (DKK/MWh):
• Observations: {len(monday_prices)}
• Mean: {monday_prices.mean():.2f}
• Std: {monday_prices.std():.2f}
• Min: {monday_prices.min():.2f}
• Max: {monday_prices.max():.2f}

Log-Transformed Series:
• Mean: {y.mean():.2f}
• Std: {y.std():.2f}
• Min: {y.min():.2f}
• Max: {y.max():.2f}

Methodology:
Weekly time-series constructed by:
1. Taking average spot price on Monday
2. Taking logs of this average
3. Multiplying by 10
        """
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show(block=True)
        
        # Alternative: Save additional plots
        # plt.savefig('nord_pool_data_analysis.png', dpi=300, bbox_inches='tight')
        # print("Analysis plots saved as 'nord_pool_data_analysis.png'")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    # Print model summary if available
    try:
        print("-"*80)
        print("MODEL SUMMARY")
        print("-"*80)
        if hasattr(model, 'summary'):
            print("Model Summary:")
            print(model.summary())
        elif hasattr(model, 'get_params'):
            print("Model Parameters:")
            print(model.get_params())
        else:
            print("Model fitted successfully - use model object for further analysis")
    except Exception as e:
        print(f"Could not display model summary: {e}")
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return model, y, monday_prices

if __name__ == "__main__":
    # Run the main analysis
    model, y, monday_prices = main()