import polars as pl
import re
from datetime import datetime
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydub import AudioSegment
from pydub.generators import Sine
import tempfile
import os
import base64

def parse_log_file(file_path):
    data = []
    
    with open(file_path, 'r') as file:
        for line in file:
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

                ######################################################
                # print(f"Full line: {line}")
                # print(f"Parts: {parts}")
                # print(f"Bid index: {bid_index}")
                # print(f"Attempting to convert: {parts[bid_index + 2]}")
                ######################################################
                
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
                
                except:
                    print(f"Bid index: {bid_index}")
                    print(f"Attempting to convert: {parts[bid_index + 2]}")
                    print(line)
    
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

def create_volatility_heatmap(df):
    unique_timestamps = df['timestamp'].unique().sort()
    
    df_call = df.filter(pl.col('option_type') == 'Call')
    df_call = df_call.with_columns(((pl.col('maturity') - pl.col('timestamp')).dt.total_seconds() / (24 * 60 * 60)).alias('days_to_expiry'))

    # Create figure
    fig = make_subplots(rows=1, cols=1, subplot_titles=["Volatility Surface"])

    # Create initial heatmap
    heatmap = go.Heatmap(
        x=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['days_to_expiry'],
        y=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['strike'],
        z=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['implied_volatility'],
        colorscale='Viridis',
        colorbar=dict(title='Implied Volatility'),
    )

    fig.add_trace(heatmap)

    # Create frames for animation
    frames = []
    for timestamp in unique_timestamps:
        df_slice = df_call.filter(pl.col('timestamp') == timestamp)
        frame = go.Frame(
            data=[go.Heatmap(
                x=df_slice['days_to_expiry'],
                y=df_slice['strike'],
                z=df_slice['implied_volatility'],
                colorscale='Viridis',
            )],
            name=str(timestamp)
        )
        frames.append(frame)

    fig.frames = frames

    # Configure the animation
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]
            )]
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[f.name], dict(mode='immediate', frame=dict(duration=500, redraw=True), transition=dict(duration=300))],
                label=f.name
            ) for f in frames],
            transition=dict(duration=300),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix='Timestamp: ', visible=True, xanchor='center'),
            len=1.0
        )]
    )

    fig.update_layout(
        title='Volatility Surface Animation',
        xaxis_title='Days to Expiration',
        yaxis_title='Strike Price',
        width=900,
        height=700
    )

    return fig

def create_chord(freq, duration_ms=500):
    chord = sum(Sine(freq * i).to_audio_segment(duration=duration_ms) for i in [1, 1.25, 1.5])
    return chord.fade_in(50).fade_out(50)

def create_volatility_heatmap(df):
    unique_timestamps = df['timestamp'].unique().sort()
    
    df_call = df.filter(pl.col('option_type') == 'Call')
    df_call = df_call.with_columns(((pl.col('maturity') - pl.col('timestamp')).dt.total_seconds() / (24 * 60 * 60)).alias('days_to_expiry'))

    # Create figure
    fig = make_subplots(rows=1, cols=1, subplot_titles=["Volatility Surface"])

    # Create initial heatmap
    heatmap = go.Heatmap(
        x=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['days_to_expiry'],
        y=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['strike'],
        z=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['implied_volatility'],
        colorscale='Viridis',
        colorbar=dict(title='Implied Volatility'),
    )

    fig.add_trace(heatmap)

    # Create frames for animation
    frames = []
    for timestamp in unique_timestamps:
        df_slice = df_call.filter(pl.col('timestamp') == timestamp)
        frame = go.Frame(
            data=[go.Heatmap(
                x=df_slice['days_to_expiry'],
                y=df_slice['strike'],
                z=df_slice['implied_volatility'],
                colorscale='Viridis',
            )],
            name=str(timestamp)
        )
        frames.append(frame)

    fig.frames = frames

    # Configure the animation
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]
            )]
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[f.name], dict(mode='immediate', frame=dict(duration=500, redraw=True), transition=dict(duration=300))],
                label=f.name
            ) for f in frames],
            transition=dict(duration=300),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix='Timestamp: ', visible=True, xanchor='center'),
            len=1.0
        )]
    )

    fig.update_layout(
        title='Volatility Surface Animation',
        xaxis_title='Days to Expiration',
        yaxis_title='Strike Price',
        width=900,
        height=700
    )

    return fig

def create_volatility_scatter3d(df):
    unique_timestamps = df['timestamp'].unique().sort()
    
    df_call = df.filter(pl.col('option_type') == 'Call')
    df_call = df_call.with_columns(((pl.col('maturity') - pl.col('timestamp')).dt.total_seconds() / (24 * 60 * 60)).alias('days_to_expiry'))

    # Create initial 3D scatter plot
    fig = go.Figure()

    scatter = go.Scatter3d(
        x=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['days_to_expiry'],
        y=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['strike'],
        z=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['implied_volatility'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_call.filter(pl.col('timestamp') == unique_timestamps[0])['implied_volatility'],
            colorscale='Viridis',
            opacity=0.8
        ),
    )

    fig.add_trace(scatter)

    # Create frames for animation
    frames = []
    for timestamp in unique_timestamps:
        df_slice = df_call.filter(pl.col('timestamp') == timestamp)
        frame = go.Frame(
            data=[go.Scatter3d(
                x=df_slice['days_to_expiry'],
                y=df_slice['strike'],
                z=df_slice['implied_volatility'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_slice['implied_volatility'],
                    colorscale='Viridis',
                    opacity=0.8
                ),
            )],
            name=str(timestamp)
        )
        frames.append(frame)

    fig.frames = frames

    # Configure the animation
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]
            )]
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[f.name], dict(mode='immediate', frame=dict(duration=500, redraw=True), transition=dict(duration=300))],
                label=f.name
            ) for f in frames],
            transition=dict(duration=300),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix='Timestamp: ', visible=True, xanchor='center'),
            len=1.0
        )]
    )

    fig.update_layout(
        title='Volatility Surface Animation (3D Scatter)',
        scene=dict(
            xaxis_title='Days to Expiration',
            yaxis_title='Strike Price',
            zaxis_title='Implied Volatility'
        ),
        width=900,
        height=700
    )

    return fig

def main(file_path, plot_type='heatmap', save_path=False):
    df = parse_log_file(file_path)
    df = process_data(df)

    print(f"Number of unique strikes: {len(df['strike'].unique())}")

    if save_path:
        df.write_excel("OptionData.xlsx")

    if plot_type == 'scatter3d':
        fig = create_volatility_scatter3d(df)
    else:
        fig = create_volatility_heatmap(df)

    fig.show()

if __name__ == "__main__":
    # file_path = "C:/Users/mbeckhusen/Documents/VolaSurfer/SPY_Options_log.txt"
    file_path = "C:/Users/mbeckhusen/Documents/VolaSurfer/SampleConsolodatedTradeBarLog.txt"
    main(file_path, plot_type='heatmap')