import polars as pl
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
    
    return pl.DataFrame(data)

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
    df = df.with_columns([
        ((pl.col('bid_open') + pl.col('bid_close')) / 2).alias('bid_mid'),
        ((pl.col('ask_open') + pl.col('ask_close')) / 2).alias('ask_mid')
    ])

    df = df.with_columns(((pl.col('bid_mid') + pl.col('ask_mid')) / 2).alias('mid_price'))

    df = df.with_columns([
        ((pl.col('maturity').cast(pl.Datetime) - pl.col('timestamp')).dt.total_seconds() / (365 * 24 * 60 * 60)).alias('T')
    ])

    df = df.with_columns([
        pl.struct(['mid_price', 'strike', 'T'])
        .map_elements(lambda x: calc_implied_volatility(x['mid_price'], x['strike'], x['T'], S, r))
        .alias('implied_volatility')
    ])

    # Filter out rows with NaN or infinite implied volatility
    df = df.filter(pl.col('implied_volatility').is_finite())

    return df

def calculate_distribution(data, column_name, num_bins=20):
    column_data = data[column_name].to_numpy()
    hist, bin_edges = np.histogram(column_data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

def plot_distributions(df, column_name, num_bins=20):
    plt.figure(figsize=(15, 10))
    
    # Group by option_type, strike, and maturity
    grouped = df.group_by(['option_type', 'strike', 'maturity']).agg(pl.all())  # Ensure the groupby step is correct

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


def main(file_path, column_name, num_bins=20):
    df = parse_log_file(file_path)
    df = process_data(df)

    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the data.")
        return

    plot_distributions(df, column_name, num_bins)

if __name__ == "__main__":
    file_path = "/Users/mikeb/Desktop/VolaSurfer/SampleConsolodatedTradeBarLog.txt"
    column_name = "implied_volatility"  # Change this to the column you want to plot
    num_bins = 20  # Change this to adjust the number of bins in the distribution
    main(file_path, column_name, num_bins)