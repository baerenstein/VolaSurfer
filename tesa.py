import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import re
from scipy.stats import norm
from scipy.optimize import brentq
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go


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
            if days <= 20:
                return 0
            elif days <= 30:
                return 1
            elif days <= 50:
                return 2
            elif days <= 70:
                return 3
            else:
                return 4
            
        # print(df.columns)

        df = df.reset_index()
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


class DataHandler:
	def __init__(self):
		self.util = Utils()
		self.modelutils = ModelUtils()

	def parse_file(self, file_path=int):

		price_data = []
		option_data = []

		# Try different encodings
		encodings = ['utf-8', 'iso-8859-1', 'cp1252']
		
		for encoding in encodings:
			try:
				with open(file_path, 'r', encoding=encoding) as file:
					for line in file:
						if 'SPY:' in line:
							parts = line.split()
							timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
							ticker = 'SPY'
				
							try:
								open_price = float(parts[4])
								high = float(parts[6])
								low = float(parts[8])
								close = float(parts[10])
								volume = float(parts[12])
				
								price_data.append({
									'timestamp': timestamp,
									'ticker': ticker,
									'open': open_price,
									'high': high,
									'low': low,
									'close': close,
									'volume': volume,
								})
				
							except:
								print("we have problem")
						
						if 'SPY' in line and ('Bid:' in line and 'Ask:' in line):
							parts = line.split()
							timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
							ticker = 'SPY'
				
							contract = parts[3].rstrip(':')
							
							match = re.match(r'(\d{6})([CP])(\d+)', contract)
							if match:
								maturity = datetime.strptime(match.group(1), "%y%m%d").date()
								option_type = 'Call' if match.group(2) == 'C' else 'Put'
								strike = float(match.group(3)) / 1000
							
							# Find indices for bid and ask data
							bid_index = parts.index('Bid:')
							ask_index = parts.index('Ask:')
				
							# Extract bid data
							try:
								bid_open = float(parts[bid_index + 2])
								bid_high = float(parts[bid_index + 5])
								bid_low = float(parts[bid_index + 8])
								bid_close = float(parts[bid_index + 11])
				
								# Extract ask data
								ask_open = float(parts[ask_index + 2])
								ask_high = float(parts[ask_index + 5])
								ask_low = float(parts[ask_index + 8])
								ask_close = float(parts[ask_index + 11])
								
								option_data.append({
									'timestamp': timestamp,
									'ticker': ticker,
									'maturity': maturity,
									'strike': strike,
									'option_type': option_type,
									'bid_open': bid_open,
									'bid_high': bid_high,
									'bid_low': bid_low,
									'bid_close': bid_close,
									'ask_open': ask_open,
									'ask_high': ask_high,
									'ask_low': ask_low,
									'ask_close': ask_close
								})       
				
							except:
								print(f"error during file parsing")
								pass
				break
			except UnicodeDecodeError:
				continue
		else:
			raise ValueError(f"Unable to decode the file with any of the attempted encodings: {encodings}")
				
		df1 = pd.DataFrame(price_data)
		df2 = pd.DataFrame(option_data)
		
		df1 = df1.set_index('timestamp')
		df2 = df2.set_index('timestamp')
		
		merged_df = df2.merge(df1, left_index=True, right_index=True, how='left', suffixes=('_option', '_price'))
		merged_df = merged_df.reset_index()
		
		return merged_df
	
	def get_basic_data(self, df=pd.DataFrame):
		"""
		Create dataframe with essential features, namely mid prices, time to maturity and implied volatility.

		Args:
			df (pd.DataFrame): pandas dataframe with option prices.

		Returns:
			df
		"""
		df['bid_mid'] = (df['bid_open'] + df['bid_close']) / 2
		df['ask_mid'] = (df['ask_open'] + df['ask_close']) / 2
		df['mid_price'] = (df['bid_mid'] + df['ask_mid']) / 2
		df['maturity'] = pd.to_datetime(df['maturity'])
		df['T'] = (df['maturity'] - df['timestamp']).dt.total_seconds() / (365 * 24 * 60 * 60)
		df['days_to_expiry'] = (df['maturity'] - df['timestamp']).dt.total_seconds() / (24 * 60 * 60)
	
		df['implied_volatility'] = df.apply(lambda row: self.util.calc_implied_volatility(
				row['option_type'],
				row['mid_price'], 
				row['strike'], 
				row['T'], 
				row['close'], 
				r=0.05), 
				axis=1)
		
		df['log_strike'] = 1 + np.log(df['strike'] / df['close'])
		
		df['vega'] = df.apply(lambda row: self.modelutils.bs_vega(
				row['close'], 
				row['strike'], 
				row['T'], 
				0.05,
				row['implied_volatility']), 
				axis=1)
		
		return df

	def get_binned_data(
			self, 
            df=pd.DataFrame,
            strike_bins=int, 
            maturity_bins=int, 
            index_lag=int, 
            weighting_method=str):
		"""_summary_

		Args:
			df (pd.DataFrame): pandas dataframe with option prices and basic features.
			bins (_type_, optional): _description_. Defaults to tuple.
			index_lag (_type_, optional): _description_. Defaults to int.
			weighting_method (_type_, optional): _description_. Defaults to str.
		"""

		df = self.util.binning(df, strike_bins) #, weighting_method
		# print(df)

		return df	

@dataclass
class OptionsData:
    symbol: str
    timestamps: List[datetime]
    expiration_dates: List[datetime]
    strikes: List[float]
    calls: pd.DataFrame
    puts: pd.DataFrame


class DataCollector:
    def __init__(self):
        self.data_handler = DataHandler()
        self.cache = {}  # Simple in-memory cache

    def collect_options_data(self, file_path: str) -> Optional[OptionsData]:
        """Collect and organize options data from the log file"""
        try:
            # Use existing DataHandler to parse the file
            df = self.data_handler.parse_file(file_path)
            df = self.data_handler.get_basic_data(df)

            # Split into calls and puts
            calls_df = df[df['option_type'] == 'Call']
            puts_df = df[df['option_type'] == 'Put']

            # Create OptionsData object
            options_data = OptionsData(
                symbol='SPY',
                timestamps=sorted(df['timestamp'].unique().tolist()),
                expiration_dates=sorted(df['maturity'].unique().tolist()),
                strikes=sorted(df['strike'].unique().tolist()),
                calls=calls_df,
                puts=puts_df
            )

            # Cache the results
            self.cache[file_path] = options_data
            
            return options_data

        except Exception as e:
            raise Exception(f"Error collecting options data: {str(e)}")

    def get_cached_data(self, file_path: str) -> Optional[OptionsData]:
        """Retrieve cached data if available, otherwise collect new data"""
        if file_path in self.cache:
            return self.cache[file_path]
        return self.collect_options_data(file_path)
    
    def get_cached_data_as_df(self, file_path: str) -> pd.DataFrame:
        """Retrieve cached data as a DataFrame indexed by timestamp"""
        options_data = self.get_cached_data(file_path)
        combined_df = pd.concat([options_data.calls, options_data.puts])
        combined_df = combined_df.set_index(['timestamp', 'days_to_expiry', 'strike', 'option_type'])
        combined_df.sort_index(inplace=True)
        return combined_df


def calculate_realized_volatility(data: OptionsData, window: int = 21) -> pd.Series:
    """
    Calculate the realized volatility of the underlying asset.
    
    Args:
        data (OptionsData): Options data from DataCollector
        window (int): Rolling window for volatility calculation
    
    Returns:
        pd.Series: Realized volatility series indexed by timestamp
    """
    # Assuming the first row of calls contains the underlying price
    underlying_prices = data.calls.groupby('timestamp')['close'].first()
    return underlying_prices.pct_change().rolling(window).std() * np.sqrt(252)

# use the DataCollector.get_cached_data_as_df() as df input
def get_nearest_contracts(df: pd.DataFrame, target_days=30, n_strikes=4):
    # Create an empty list to store filtered dataframes
    filtered_dfs = []
    
    # Process each unique timestamp
    for timestamp in df.index.get_level_values('timestamp').unique():
        # Get data for this timestamp
        df_t = df.xs(timestamp, level='timestamp')
        
        # Find the expiry closest to target_days
        days_to_expiry = df_t.index.get_level_values('days_to_expiry').unique()
        nearest_expiry = days_to_expiry[np.abs(days_to_expiry - target_days).argmin()]
        
        # Get data for nearest expiry
        df_te = df_t.xs(nearest_expiry, level='days_to_expiry')
        
        # Get the close price for this timestamp
        close_price = df_te['close'].iloc[0]
        
        # Get unique strikes
        strikes = df_te.index.get_level_values('strike').unique()
        
        # Find n closest strikes to close price
        closest_strikes = strikes[np.argsort(np.abs(strikes - close_price))[:n_strikes]]
        
        # Filter for these strikes
        df_tes = df_te[df_te.index.get_level_values('strike').isin(closest_strikes)]
        
        # Add back the timestamp index level
        df_tes = df_tes.assign(timestamp=timestamp).set_index('timestamp', append=True).reorder_levels(['timestamp', 'strike', 'option_type'])
        
        filtered_dfs.append(df_tes)
    
    # Combine all filtered dataframes
    result = pd.concat(filtered_dfs)
    
    return result

def calculate_iv_index(contracts_df: pd.DataFrame) -> pd.Series:
    """
    Calculate vega-weighted implied volatility index from nearest contracts.
    For each timestamp, aggregates implied volatilities weighted by vega separately 
    for calls and puts, then averages the two.
    
    Args:
        contracts_df: DataFrame with multi-index [timestamp, strike, option_type]
                     containing 'implied_vol' and 'vega' columns
    
    Returns:
        Series indexed by timestamp containing the IV index values
    """
    # Initialize list to store results
    iv_index_values = []
    timestamps = contracts_df.index.get_level_values('timestamp').unique()
    
    for ts in timestamps:
        ts_data = contracts_df.xs(ts, level='timestamp')
        
        # Calculate weighted IV separately for calls and puts
        call_data = ts_data.xs('Call', level='option_type')
        put_data = ts_data.xs('Put', level='option_type')
        
        # Weight IVs by vega
        weighted_call_iv = (call_data['implied_volatility'] * call_data['vega']).sum() / call_data['vega'].sum()
        weighted_put_iv = (put_data['implied_volatility'] * put_data['vega']).sum() / put_data['vega'].sum()
        
        # Average of call and put weighted IVs
        iv_index = (weighted_call_iv + weighted_put_iv) / 2
        
        iv_index_values.append((ts, iv_index))
    
    # Create series from results
    iv_index_series = pd.Series(dict(iv_index_values), name='iv_index')
    iv_index_series.index.name = 'timestamp'
    
    return iv_index_series

def create_iv_index_histogram(iv_index: pd.Series, bins=100):

    iv_index = iv_index.dropna()

    last_iv = iv_index.iloc[-1]
    iv_95_percentile = np.percentile(iv_index, 95)
    iv_5_percentile = np.percentile(iv_index, 5)

    iv_index.hist(bins=bins)
    plt.title("IV Index Distribution")
    plt.axvspan(last_iv - 0.01, last_iv + 0.01, color='yellow', alpha=0.3, label="Last Value")
    # Add arrow annotation for the last value
    plt.annotate(f'Last Value: {last_iv:.2f}', xy=(last_iv, 0), xytext=(last_iv + 0.5, 5), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
    # Add vertical lines for percentiles
    plt.axvline(iv_95_percentile, color='orange', linestyle='--', label=f'95th Percentile: {iv_95_percentile:.2f}')
    plt.axvline(iv_5_percentile, color='purple', linestyle='--', label=f'5th Percentile: {iv_5_percentile:.2f}')
    

    plt.show()

#TODO iv percentage change series/plot

#TODO vol of vol series/plot

#TODO fit to rest of the code
def create_smile_plot(self, df, timestamp):
    """Create volatility smile plot for specific timestamp"""
    try:
        df_slice = df[df['timestamp'] == timestamp]
        fig = go.Figure()
        for option_type in ['Call', 'Put']:
            options = df_slice[df_slice['option_type'] == option_type]
            fig.add_trace(go.Scatter(
                x=options['log_strike'],
                y=options['implied_volatility'],
                mode='markers+lines',
                name=option_type,
                marker=dict(size=8)
            ))
        fig.update_layout(
            title=f"Volatility Smile at {timestamp}",
            xaxis_title="Log Strike",
            yaxis_title="Implied Volatility",
            showlegend=True
        )
        return fig
    except Exception as e:
        self.logger.error(f"Error creating smile plot: {str(e)}")
        return go.Figure()

#TODO fit to rest of the code
def create_term_structure_plot(self, df, timestamp):
    """Create volatility term structure plot"""
    try:
        df_slice = df[df['timestamp'] == timestamp]
        atm_options = df_slice[abs(df_slice['log_strike'] - 1) < 0.05]
        term_structure = atm_options.groupby('days_to_expiry')['implied_volatility'].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=term_structure.index,
            y=term_structure.values,
            mode='markers+lines',
            name='ATM Term Structure'
        ))
        fig.update_layout(
            title=f"ATM Volatility Term Structure at {timestamp}",
            xaxis_title="Days to Expiry",
            yaxis_title="ATM Implied Volatility"
        )
        return fig
    except Exception as e:
        self.logger.error(f"Error creating term structure plot: {str(e)}")
        return go.Figure()

#TODO currently returns only bins available from the options data, hence there are gaps. change so that a grid is created
def create_volatility_heatmap(df: pd.DataFrame):
    timestamps = df['timestamp'].unique()

    for ts in timestamps:
        # Filter data for current timestamp
        ts_data = df[df['timestamp'] == ts]
        
        # Create pivot table for heatmap
        pivot_data = ts_data.pivot_table(
            values='implied_volatility',
            index='log_strike_bin',
            columns='expiry_bin',
            aggfunc='mean'
        )
    colors = ['#2ecc71',  # bright green
            '#a8e6cf',  # light green
            '#f1f6a6',  # light yellow
            '#ffd3b6',  # light orange
            '#ffaaa5',  # light red
            '#ff8b94',  # medium red
            '#e74c3c']  # bright red
    custom_cmap = LinearSegmentedColormap.from_list('custom_green_red', colors)

    # Calculate global min and max for consistent color scaling
    vmin = df['implied_volatility'].min()
    vmax = df['implied_volatility'].max()

    timestamps = df['timestamp'].unique()

    for ts in timestamps:
        # Filter data for current timestamp
        ts_data = df[df['timestamp'] == ts]
        
        # Create pivot table for heatmap
        pivot_data = ts_data.pivot_table(
            values='implied_volatility',
            index='log_strike_bin',
            columns='expiry_bin',
            aggfunc='mean'
        )

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(
            pivot_data,
            cmap=custom_cmap,
            annot=True,  # Show values in cells
            fmt='.3f',   # Format for the annotations
            cbar_kws={'label': 'Implied Volatility'},
            square=True,
            vmin=vmin,
            vmax=vmax
        )

        # Customize the plot
        plt.title(f'Implied Volatility Surface at {ts}\nVol Range: [{vmin:.3f}, {vmax:.3f}]')
        plt.xlabel('Expiry Bins')
        plt.ylabel('Log Strike Bins')

        # Add text explaining the bins
        bin_explanation = (
            "Expiry Bins:\n"
            "0: ≤ 25 days\n"
            "1: 26-50 days\n"
            "2: 51-75 days\n"
            "3: > 75 days"
        )
        plt.figtext(1.15, 0.6, bin_explanation, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

        # Adjust layout to prevent cutting off labels
        plt.tight_layout()
        plt.show()

#TODO merge with above
def create_volatility_heatmap2(self, df, bins=10):
    """Create volatility heatmap using binned implied volatilities"""
    try:
        df['log_strike_bin'] = pd.cut(df['log_strike'], bins=bins)
        df['days_to_expiry_bin'] = pd.cut(df['days_to_expiry'], bins=bins)
        pivot = df.pivot_table(
            values='implied_volatility',
            index='log_strike_bin',
            columns='days_to_expiry_bin',
            aggfunc='mean'
        )
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorscale='Viridis',
            colorbar=dict(title='Implied Volatility')
        ))
        fig.update_layout(
            title="Implied Volatility Heatmap",
            xaxis_title="Days to Expiry",
            yaxis_title="Log Strike",
            xaxis_tickangle=-45
        )
        return fig
    except Exception as e:
        self.logger.error(f"Error creating volatility heatmap: {str(e)}")
        return go.Figure()

# def create_volatility_surface(data: OptionsData) -> pd.DataFrame:
#     """
#     Create a volatility surface from options data.
    
#     Args:
#         data (OptionsData): Options data from DataCollector
    
#     Returns:
#         pd.DataFrame: Volatility surface with strikes as rows and days to expiry as columns
#     """
#     # Combine calls and puts
#     df = pd.concat([
#         data.calls[['strike', 'days_to_expiry', 'implied_volatility']],
#         data.puts[['strike', 'days_to_expiry', 'implied_volatility']]
#     ])
    
#     # Create pivot table for the surface
#     surface = df.pivot_table(
#         values='implied_volatility',
#         index='strike',
#         columns='days_to_expiry',
#         aggfunc='mean'
#     )
#     return surface

#TODO adjust axis descriptions
def create_volatility_surface(df, timestamp=None):
    """Create 3D volatility surface visualization"""
    
    try:
        # Reset index and set appropriate indexes
        df.reset_index(inplace=True)
        df.set_index(['timestamp', 'option_type', 'days_to_expiry', 'log_strike'], inplace=True)
        
        # Filter for the latest timestamp if not specified
        if timestamp is None:
            latest_timestamp = df.index.get_level_values('timestamp').max()
            df = df[df.index.get_level_values('timestamp') == latest_timestamp]
        else:
            df = df[df.index.get_level_values('timestamp') == timestamp]
        
        df_calls = df[df.index.get_level_values('option_type') == 'Call']
        
        #####
        # Get the close price for the ATM calculation
        atm_price = df_calls['close'].iloc[0]  # Assuming the first row has the ATM price
        
        # Calculate the ATM strike
        atm_strike = df_calls['strike'].iloc[(df_calls['strike'] - atm_price).abs().argsort()[:1]].values[0]
        
        # Adjust log strikes to center around ATM strike
        df_calls['log_strike_centered'] = np.log(df_calls['strike'] / atm_strike)
        #####
    
        # Pivot to create a 2D grid for the surface
        pivot_calls = df_calls.pivot_table(
            values='implied_volatility',
            index='days_to_expiry',
            # columns='log_strike',
            columns='log_strike_centered',
            aggfunc='mean'
        )
        
        # Ensure pivot table is not empty
        if pivot_calls.empty:
            raise ValueError("Pivot table is empty, no data to plot.")
        
        surface = go.Figure(data=[go.Surface(
            x=pivot_calls.columns,
            y=pivot_calls.index,
            z=pivot_calls.values,
            colorscale='rdbu',
            opacity=0.8
        )])
        title = "Raw Implied Volatility Surface"

        surface.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Days to Expiry",
                yaxis_title="Log Strike",
                zaxis_title="Implied Volatility",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=65, r=50, b=65, t=90)
        )
        surface.show()
        return surface
    except Exception as e:
        print(f"Error creating volatility surface: {str(e)}")
        return go.Figure()

#####################################################################
#---------------- testing classes, methods and functions -----------#
#####################################################################

# get data
dc = DataCollector()
dc.collect_options_data("/Users/mikeb/Desktop/VolaSurfer/backend/core/data/SPY_Options_log.txt")
options_data = dc.get_cached_data("/Users/mikeb/Desktop/VolaSurfer/backend/core/data/SPY_Options_log.txt")
df = dc.get_cached_data_as_df("/Users/mikeb/Desktop/VolaSurfer/backend/core/data/SPY_Options_log.txt") 

# get realized volatility
realized_vol = calculate_realized_volatility(options_data) #checked
# print(type(realized_vol))

# get iv index
iv_index_series = calculate_iv_index(df) #checked
# print(type(iv_index_series))
# print(iv_index_series)

# dh = DataHandler() #checked
# binned_df = dh.get_binned_data(df, strike_bins=6) #checked

# create_volatility_heatmap(binned_df) #check

# create_iv_index_histogram(iv_index_series, bins=30) #check


create_volatility_surface(df) #check

#TODO from rolling_vol_smile_heat, smile over time animation