import polars as pl
import pandas as pd
import re
from datetime import datetime
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def parse_log_file(file_path):
    data = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if 'SPY' in line and ('Bid:' in line and 'Ask:' in line):
                parts = line.split()
                timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
                ticker = 'SPY'
                
                bid_index = parts.index('Bid:')
                ask_index = parts.index('Ask:')
                
                contract = ''.join(parts[3:bid_index])
                
                match = re.match(r'(\d{6})([CP])(\d+)', contract)
                if match:
                    maturity = datetime.strptime(match.group(1), "%y%m%d").date()
                    option_type = 'Call' if match.group(2) == 'C' else 'Put'
                    strike = float(match.group(3)) / 1000
                
                bid_open, bid_high, bid_low, bid_close = map(float, [parts[bid_index + i] for i in [2, 5, 8, 11]])
                ask_open, ask_high, ask_low, ask_close = map(float, [parts[ask_index + i] for i in [2, 5, 8, 11]])
                
                data.append({
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
    
    df = pd.DataFrame(data)
    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(price, S, K, T, r):
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - price
    return brentq(objective, 1e-6, 10)

def calc_implied_volatility(mid_price, strike, T, S=170, r=0.05):
    try:
        return implied_volatility(mid_price, S, strike, T, r)
    except:
        return None

def process_data(df, S=170, r=0.05):
    # Calculate bid_mid and ask_mid
    df['bid_mid'] = (df['bid_open'] + df['bid_close']) / 2
    df['ask_mid'] = (df['ask_open'] + df['ask_close']) / 2

    # Calculate mid_price as the average of bid_mid and ask_mid
    df['mid_price'] = (df['bid_mid'] + df['ask_mid']) / 2

    # Calculate time to maturity (T) in years
    df['T'] = (df['maturity'] - df['timestamp'].dt.date).apply(lambda x: x.days) / 365

    # Calculate implied volatility for each row
    df['implied_volatility'] = df.apply(lambda row: calc_implied_volatility(row['mid_price'], row['strike'], row['T'], S, r), axis=1)

    # Filter out rows with NaN or infinite implied volatility
    df = df[df['implied_volatility'].notna() & np.isfinite(df['implied_volatility'])]

    return df


def calculate_distribution(data, column_name, num_bins=20):
    column_data = data[column_name].to_numpy()
    hist, bin_edges = np.histogram(column_data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

def plot_distributions(df, column_name, num_bins=20):
    plt.figure(figsize=(15, 10))
    
    # Group by option_type, strike, and maturity
    grouped = df.groupby(['option_type', 'strike', 'maturity']).agg(pl.all())  # Ensure the groupby step is correct

    # Create a custom colormap using the number of unique groups
    num_groups = grouped.shape[0]  # Get the number of groups
    colors = plt.cm.viridis(np.linspace(0, 1, num_groups))
    
    max_hist = 0
    min_bin = float('inf')
    max_bin = float('-inf')

    # Iterate over each group row-wise
    for i, group in enumerate(grouped.rows()):
        option_type, strike, maturity = group[0], group[1], group[2]  # Extract the values from the group
        
        # Create a new dataframe from the group values
        group_df = df.filter(
            (df['option_type'] == option_type) & (df['strike'] == strike) & (df['maturity'] == maturity)
        )

        bin_centers, hist = calculate_distribution(group_df, column_name, num_bins)
        
        # Update min and max bin values
        min_bin = min(min_bin, bin_centers.min())
        max_bin = max(max_bin, bin_centers.max())
        
        # Update max histogram value
        max_hist = max(max_hist, hist.max())
        
        # Plot the distribution
        plt.fill_between(bin_centers, i, i + hist / max_hist, alpha=0.8, color=colors[i])
        plt.plot(bin_centers, i + hist / max_hist, color='k', alpha=0.5)
        
        # Add label
        plt.text(max_bin, i + 0.25, f"{option_type}, K: {strike:.2f}, T: {maturity}", 
                 fontsize=8, verticalalignment='center')

    plt.ylim(0, num_groups)
    plt.xlim(min_bin, max_bin)
    
    plt.title(f"Distribution of {column_name} for Different Contracts")
    plt.xlabel(column_name)
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_volatility_smile(df, column_name, num_bins=20):
    plt.figure(figsize=(15, 10))

    # Sort by the 'timestamp' column to ensure time is properly ordered
    df = df.sort_values('timestamp')

    # Group by strike, maturity, and option_type (i.e., each unique option contract)
    grouped = df.groupby(['strike', 'maturity', 'option_type'])

    num_groups = grouped.ngroups  # Number of unique option contracts
    colors = plt.cm.viridis(np.linspace(0, 1, num_groups))
    
    min_vol = float('inf')
    max_vol = float('-inf')

    # Iterate over each group (by unique option contract)
    for i, (name, group_df) in enumerate(grouped):
        strike, maturity, option_type = name

        # Sort by timestamp within each group to keep time ordered
        group_df = group_df.sort_values('timestamp')

        # Get the time in days for the y-axis
        y_vals = (group_df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / (24 * 60 * 60)  # Time in days
        
        # Get the implied volatilities for the x-axis
        x_vals = group_df[column_name]

        # Update min and max values for implied volatility (x-axis)
        min_vol = min(min_vol, x_vals.min())
        max_vol = max(max_vol, x_vals.max())

        # Plot the implied volatilities for each option contract over time
        plt.plot(x_vals, y_vals, color=colors[i], label=f'Strike: {strike}, {option_type}')

    plt.xlim(min_vol, max_vol)
    
    # Set the y-axis to the range of days (time)
    plt.ylim(0, (df['timestamp'] - df['timestamp'].min()).dt.total_seconds().max() / (24 * 60 * 60))

    plt.title(f"Implied Volatility Skew/Smile Over Time for Each Contract")
    plt.xlabel(column_name)
    plt.ylabel("Time (Days)")
    plt.tight_layout()
    plt.legend()
    plt.show()

def main(file_path, column_name, num_bins=20):
    df = parse_log_file(file_path)
    df = process_data(df)

    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the data.")
        return

    plot_volatility_smile(df, column_name, num_bins)

if __name__ == "__main__":
    file_path = "/Users/mikeb/Desktop/VolaSurfer/SampleConsolodatedTradeBarLog.txt"
    column_name = "implied_volatility"  # Change this to the column you want to plot
    num_bins = 20  # Change this to adjust the number of bins in the distribution
    main(file_path, column_name, num_bins)