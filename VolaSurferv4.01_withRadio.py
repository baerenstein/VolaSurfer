import polars as pl
import re
from datetime import datetime
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import time
import sounddevice as sd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

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

def iv_to_frequency(iv, min_iv, max_iv, min_freq=100, max_freq=10000):
    return min_freq * (max_freq / min_freq) ** ((iv - min_iv) / (max_iv - min_iv))

def generate_tone(frequency, duration=0.1, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    return tone

def play_tone(tone, sample_rate=44100):
    sd.play(tone, sample_rate)
    sd.wait()

def plot_volatility_surface(ax, df_slice):
    df_call = df_slice.filter(pl.col('option_type') == 'Call')
    
    # Convert maturity to numeric (days to expiration)
    df_call = df_call.with_columns(((pl.col('maturity') - pl.col('timestamp')).dt.total_seconds() / (24 * 60 * 60)).alias('days_to_expiry'))
    
    X = df_call['days_to_expiry'].to_numpy()
    Y = df_call['strike'].to_numpy()
    Z = df_call['implied_volatility'].to_numpy()

    ax.clear()
    ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Days to Expiration')
    ax.set_ylabel('Strike Price')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Volatility Surface ({df_slice["timestamp"].unique()[0]})')

def sonify_and_plot_volatility_surface(df, timestamp, ax, fig):
    df_slice = df.filter(pl.col('timestamp') == timestamp)
    df_call = df_slice.filter(pl.col('option_type') == 'Call')
    
    ivs = df_call['implied_volatility'].to_numpy()
    min_iv, max_iv = np.min(ivs), np.max(ivs)
    
    frequencies = [iv_to_frequency(iv, min_iv, max_iv) for iv in ivs]
    
    plot_volatility_surface(ax, df_slice)
    fig.canvas.draw()
    plt.pause(0.01)  # Small pause to update the plot
    
    for freq in frequencies:
        tone = generate_tone(freq)
        play_tone(tone)
        time.sleep(0.0)  # Short pause between tones

def main(file_path, save_path=None):
    df = parse_log_file(file_path)
    df = process_data(df)

    if save_path:
        df.write_csv(save_path)
        print(f"Data saved to {save_path}")

    unique_timestamps = df['timestamp'].unique()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for timestamp in unique_timestamps:
        print(f"Sonifying and plotting data for {timestamp}")
        sonify_and_plot_volatility_surface(df, timestamp, ax, fig)
        time.sleep(0.5)  # Pause between timestamps
    
    plt.show()

if __name__ == "__main__":
    file_path = "/Users/mikeb/Desktop/VolaSurfer/SampleConsolodatedTradeBarLog.txt"
    # save_path = "/path/to/your/processed_options_data.csv"  # Optional
    main(file_path)