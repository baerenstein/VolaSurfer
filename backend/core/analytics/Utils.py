import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

class Utils:

    def black_scholes_call(self, S, K, r, T, sigma):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def black_scholes_put(S: float, K: float, r: float, T: float, sigma: float):
        """
        Calculate Black-Scholes put option price and Greeks
        
        Parameters:
        -----------
        S : float
            Current stock/underlying price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annual, expressed as decimal)
        sigma : float
            Volatility (annual, expressed as decimal)
            
        Returns:
        --------
        dict
            Dictionary containing:
            - price: Put option price
            - delta: First derivative with respect to underlying price
            - gamma: Second derivative with respect to underlying price
            - theta: First derivative with respect to time
            - vega: First derivative with respect to volatility
            - rho: First derivative with respect to interest rate
        """
        # Handle edge cases
        if T <= 0:
            return max(K - S, 0)
        
        # Calculate standard Black-Scholes parameters
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt_T)
        d2 = d1 - sigma*sqrt_T
        
        # Calculate put option price
        N_d1 = norm.cdf(-d1)
        N_d2 = norm.cdf(-d2)
        put_value = K*np.exp(-r*T)*N_d2 - S*N_d1
        
        # Calculate Greeks
        n_d1 = norm.pdf(d1)
        
        delta = -N_d1
        gamma = n_d1/(S*sigma*sqrt_T)
        theta = -(S*sigma*n_d1)/(2*sqrt_T) + r*K*np.exp(-r*T)*N_d2
        vega = S*sqrt_T*n_d1
        rho = -K*T*np.exp(-r*T)*N_d2

        return put_value        
        # return {
        #     'price': put_value,
        #     'delta': delta,
        #     'gamma': gamma,
        #     'theta': theta/365,  # Convert to daily theta
        #     'vega': vega/100,   # Convert to 1% vol change
        #     'rho': rho/100      # Convert to 1% rate change
        # }
        
    def implied_volatility(self, option_type, price, S, K, r, T):
        if option_type == "Call":
            def objective(sigma):
                return self.black_scholes_call(S, K, r, T, sigma) - price
        else:
            def objective(sigma):
                return self.black_scholes_call(S, K, r, T, sigma) - price
            
        return brentq(objective, 1e-6, 10)
        
    def calc_implied_volatility(self, option_type, mid_price, strike, T, S, r=0.05):
        try:
            return self.implied_volatility(option_type, mid_price, S, strike, r, T)
        except:
            return None

    #TODO validate formulas		
    def binning(self, df=pd.DataFrame, strike_bins=int): # maturity_bins
        if 'log_strike' not in df.columns or 'implied_volatility' not in df.columns:
            raise ValueError("DataFrame must contain 'log_strike' and 'implied_volatility' columns")
        
        log_strike_min = df['log_strike'].min()
        log_strike_max = df['log_strike'].max()
        bin_edges = np.linspace(log_strike_min, log_strike_max, strike_bins + 1)

        df['log_strike_bin'] = pd.cut(
                                df['log_strike'], 
                                bins=bin_edges, 
                                labels=False, 
                                include_lowest=True
                                )

        def bin_expiry(days):
            if days <= 30:
                return 0
            elif days <= 60:
                return 1
            else:
                return 2
            
        df['expiry_bin'] = df['days_to_expiry'].apply(bin_expiry)


        
        #TODO aggregate implied vol by vega, hence calculate greeks before binning
        df = df.groupby([
                'timestamp', 
                'expiry_bin', 
                'log_strike_bin', 
                'option_type']).agg({
                    'log_strike': 'mean',
                    'implied_volatility': 'mean',
                    'days_to_expiry': 'mean'
                    }).reset_index()

        df = df.rename(columns={'log_strike': 'binned_log_strike'})
        
        return df	
    
    #TODO write vega weightde aggregation method
    def aggregate_by_vega(data):
        options_vega = int
        size_of_bin = tuple
        return

    #TODO adjust to be generic bin analyser
    def analyze_bin_quality(self, binned_df: pd.DataFrame) -> dict:
        """
        Analyze the quality of the binning by checking vega distribution.
        
        Args:
            binned_df: DataFrame with binned options data
            
        Returns:
            Dictionary with bin quality metrics
        """
        quality_metrics = {
            'total_vega': binned_df['total_vega'].sum(),
            'vega_per_bin': binned_df['total_vega'].mean(),
            'vega_std': binned_df['total_vega'].std(),
            'empty_bins': (binned_df['total_vega'] == 0).sum(),
            'avg_options_per_bin': binned_df['n_options'].mean(),
            'max_vega_concentration': (binned_df['total_vega'].max() / 
                                     binned_df['total_vega'].sum())
        }
        
        return quality_metrics

    
class ModelUtils:

    def g(self, x, a):
        """
        TBSS kernel applicable to the rBergomi variance process.
        """
        return x**a

    def b(self, k, a):
        """
        Optimal discretisation of TBSS process for minimising hybrid scheme error.
        """
        return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

    def cov(self, a, n):
        """
        Covariance matrix for given alpha and n, assuming kappa = 1 for
        tractability.
        """
        cov = np.array([[0.,0.],[0.,0.]])
        cov[0,0] = 1./n
        cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
        cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
        cov[1,0] = cov[0,1]
        return cov

    def bs(self, F, K, V, o = 'call'):
        """
        Returns the Black call price for given forward, strike and integrated
        variance.
        """
        # Set appropriate weight for option token o
        w = 1
        if o == 'put':
            w = -1
        elif o == 'otm':
            w = 2 * (K > 1.0) - 1

        sv = np.sqrt(V)
        d1 = np.log(F/K) / sv + 0.5 * sv
        d2 = d1 - sv
        P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
        return P

    def bs_vega(self, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5 * np.power(sigma, 2)) * T) / sigma *np.sqrt(T)
        vg = S * norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
        vega = np.maximum(vg, 1e-19)
        return vega

    def black_scholes_call(self, sigma, S, K, r, T):
        d1 = 1 / (sigma * np.sqrt(T)) * ( np.log(S/K) + (r + np.power(sigma,2)/2) * T)
        d2 = d1 - sigma * np.sqrt(T)
        C = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
        return C


    def find_vol(self, target_value, S, K, T, r):
        # target value: price of the option contract
        MAX_iterations = 10000
        prec = 1.0e-8
        sigma = 0.5
        for i in range(0, MAX_iterations):
            price = self.black_scholes_call(sigma, S, K, r, T)
            diff = target_value - price # the root
            if abs(diff) < prec:
                return sigma
            vega = self.bs_vega(S, K, T, r, sigma)
            sigma = sigma + diff/vega # f(x) / f'(x)
            if sigma > 10 or sigma < 0:
                sigma=0.5
            
        return sigma
        
    def bsinv(self, P, F, K, t, o = 'call'):
        """
        Returns implied Black vol from given call price, forward, strike and time
        to maturity.
        """
        # Set appropriate weight for option token o
        w = 1
        if o == 'put':
            w = -1
        elif o == 'otm':
            w = 2 * (K > 1.0) - 1

        # Ensure at least instrinsic value
        P = np.maximum(P, np.maximum(w * (F - K), 0))

        def error(s):
            return self.bs(F, K, s**2 * t, o) - P
        
        s = brentq(error, 1e-19, 1e+9)

        return s