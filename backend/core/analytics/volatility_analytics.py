from typing import Optional
import pandas as pd
import numpy as np

from .data.DataCollector import OptionsData

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

def get_implied_volatility(data: OptionsData) -> pd.Series:
    """
    Get the implied volatility of the underlying asset.
    
    Args:
        data (OptionsData): Options data from DataCollector
    
    Returns:
        pd.Series: Implied volatility series indexed by timestamp
    """
    return data.calls['implied_volatility']

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

def create_volatility_surface(data: OptionsData) -> pd.DataFrame:
    """
    Create a volatility surface from options data.
    
    Args:
        data (OptionsData): Options data from DataCollector
    
    Returns:
        pd.DataFrame: Volatility surface with strikes as rows and days to expiry as columns
    """
    # Combine calls and puts
    df = pd.concat([
        data.calls[['strike', 'days_to_expiry', 'implied_volatility']],
        data.puts[['strike', 'days_to_expiry', 'implied_volatility']]
    ])
    
    # Create pivot table for the surface
    surface = df.pivot_table(
        values='implied_volatility',
        index='strike',
        columns='days_to_expiry',
        aggfunc='mean'
    )
    return surface

def get_volatility_smile(data: OptionsData, days_to_expiry: Optional[int] = None) -> pd.DataFrame:
    """
    Get the volatility smile for a specific expiry.
    
    Args:
        data (OptionsData): Options data from DataCollector
        days_to_expiry (int, optional): Specific days to expiry to analyze
    
    Returns:
        pd.DataFrame: Volatility smile data with strikes and corresponding IVs
    """
    df = pd.concat([data.calls, data.puts])
    
    if days_to_expiry is not None:
        df = df[df['days_to_expiry'] == days_to_expiry]
    
    return df.groupby('strike')['implied_volatility'].mean()

def get_volatility_term_structure(data: OptionsData) -> pd.DataFrame:
    """
    Get the volatility term structure (IV vs time to expiry).
    
    Args:
        data (OptionsData): Options data from DataCollector
    
    Returns:
        pd.DataFrame: Term structure data with days to expiry and corresponding IVs
    """
    df = pd.concat([data.calls, data.puts])
    return df.groupby('days_to_expiry')['implied_volatility'].mean()



