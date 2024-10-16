import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq

def parse_log_file(file_path):
    price_data = []
    option_data = []

    with open(file_path, 'r') as file:
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
                    print(f"Bid index: {bid_index}")
                    print(f"Attempting to convert: {parts[bid_index + 2]}")
                    print(line)

    df1 = pd.DataFrame(price_data)
    df2 = pd.DataFrame(option_data)

    df1 = df1.set_index('timestamp')
    df2 = df2.set_index('timestamp')

    merged_df = df2.merge(df1, left_index=True, right_index=True, how='left', suffixes=('_option', '_price'))
    merged_df = merged_df.reset_index()

    return merged_df

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(price, S, K, T, r):
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - price
    return brentq(objective, 1e-6, 10)

def calc_implied_volatility(mid_price, strike, T, S, r=0.05):
    try:
        return implied_volatility(mid_price, S, strike, T, r)
    except:
        return None

def process_data(df, r=0.05):
    df['bid_mid'] = (df['bid_open'] + df['bid_close']) / 2
    df['ask_mid'] = (df['ask_open'] + df['ask_close']) / 2
    df['mid_price'] = (df['bid_mid'] + df['ask_mid']) / 2

    df['maturity'] = pd.to_datetime(df['maturity'])
    df['T'] = (df['maturity'] - df['timestamp']).dt.total_seconds() / (365 * 24 * 60 * 60)

    df['implied_volatility'] = df.apply(lambda row: calc_implied_volatility(
        row['mid_price'], row['strike'], row['T'], row['close'], r
        ), axis=1)

    df['log_strike'] = 1 + np.log(df['strike'] / df['close'])

    return df.dropna(subset=['implied_volatility'])

def create_volatility_heatmap(df):
    unique_timestamps = df['timestamp'].unique().to_numpy()
    unique_timestamps.sort()
    
    df_call = df[df['option_type'] == 'Call']
    df_call['days_to_expiry'] = (df_call['maturity'] - df_call['timestamp']).dt.total_seconds() / (24 * 60 * 60)

    fig = make_subplots(rows=1, cols=1, subplot_titles=["Volatility Surface"])

    heatmap = go.Heatmap(
        x=df_call[df_call['timestamp'] == unique_timestamps[0]]['days_to_expiry'],
        y=df_call[df_call['timestamp'] == unique_timestamps[0]]['log_strike'],
        z=df_call[df_call['timestamp'] == unique_timestamps[0]]['implied_volatility'],
        colorscale='Viridis',
        colorbar=dict(title='Implied Volatility'),
    )

    fig.add_trace(heatmap)

    frames = []
    for timestamp in unique_timestamps:
        df_slice = df_call[df_call['timestamp'] == timestamp]
        frame = go.Frame(
            data=[go.Heatmap(
                x=df_slice['days_to_expiry'],
                y=df_slice['log_strike'],
                z=df_slice['implied_volatility'],
                colorscale='Viridis',
            )],
            name=str(timestamp)
        )
        frames.append(frame)

    fig.frames = frames

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
        yaxis_title='Log Strike',
        width=900,
        height=700
    )

    return fig

def create_volatility_scatter3d(df):
    unique_timestamps = df['timestamp'].unique().to_numpy()
    unique_timestamps.sort()
    
    df_call = df[df['option_type'] == 'Call']
    df_call['days_to_expiry'] = (df_call['maturity'] - df_call['timestamp']).dt.total_seconds() / (24 * 60 * 60)

    fig = go.Figure()

    scatter = go.Scatter3d(
        x=df_call[df_call['timestamp'] == unique_timestamps[0]]['days_to_expiry'],
        y=df_call[df_call['timestamp'] == unique_timestamps[0]]['log_strike'],
        z=df_call[df_call['timestamp'] == unique_timestamps[0]]['implied_volatility'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_call[df_call['timestamp'] == unique_timestamps[0]]['implied_volatility'],
            colorscale='Viridis',
            opacity=0.8
        ),
    )

    fig.add_trace(scatter)

    frames = []
    for timestamp in unique_timestamps:
        df_slice = df_call[df_call['timestamp'] == timestamp]
        frame = go.Frame(
            data=[go.Scatter3d(
                x=df_slice['days_to_expiry'],
                y=df_slice['log_strike'],
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
            yaxis_title='Log Strike',
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
        df.to_excel("OptionData.xlsx")

    if plot_type == 'scatter3d':
        fig = create_volatility_scatter3d(df)
    else:
        fig = create_volatility_heatmap(df)

    fig.show()

if __name__ == "__main__":
    file_path = "C:/Users/mbeckhusen/Documents/VolaSurfer/SPY_Options_log.txt"
    # file_path = "C:/Users/mbeckhusen/Documents/VolaSurfer/SampleConsolodatedTradeBarLog.txt"
    main(file_path, plot_type='heatmap')