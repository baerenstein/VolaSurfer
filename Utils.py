import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

class Utils:

    def black_scholes_call(self, S, K, r, T, sigma):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
    def implied_volatility(self, price, S, K, r, T):
        def objective(sigma):
            return self.black_scholes_call(S, K, r, T, sigma) - price
        return brentq(objective, 1e-6, 10)
        
    def calc_implied_volatility(self, mid_price, strike, T, S, r=0.05):
        try:
            return self.implied_volatility(mid_price, S, strike, r, T)
        except:
            return None
			
    def binning(self, df=pd.DataFrame, days_bin=tuple, strike_bin=int):
        if 'log_strike' not in df.columns or 'implied_volatility' not in df.columns:
            raise ValueError("DataFrame must contain 'log_strike' and 'implied_volatility' columns")
        
        log_strike_min = df['log_strike'].min()
        log_strike_max = df['log_strike'].max()
        bin_edges = np.linspace(log_strike_min, log_strike_max, strike_bin + 1)
        df['log_strike_bin'] = pd.cut(
                                df['log_strike'], 
                                bins=bin_edges, 
                                labels=False, 
                                include_lowest=True
                                )
        def bin_expiry(days):
            if days <= 25:
                return 0
            elif days <= 50:
                return 1
            elif days <= 75:
                return 2
            else:
                return 3
            
        df['expiry_bin'] = df['days_to_expiry'].apply(bin_expiry)

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

    def iv_index(self, df, lag, method):
        return 