import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

def create_vol_grid(strikes, maturities, vols):
    """
    Creates a properly formatted grid for surface plotting
    
    Parameters:
    strikes: Array of unique strike prices
    maturities: Array of unique maturities (in days)
    vols: List of volatilities with corresponding strike/maturity pairs
    
    Returns:
    K, T: Meshgrid arrays
    vol_matrix: 2D array of volatilities
    """
    # Create empty grid
    vol_matrix = np.zeros((len(maturities), len(strikes)))
    
    # Fill grid with synthetic data (replace this with real data)
    for i, t in enumerate(maturities):
        for j, k in enumerate(strikes):
            # Synthetic smile effect
            moneyness = np.log(k/100)
            vol_matrix[i,j] = 0.2 + 0.05 * moneyness**2 + 0.02 * np.sqrt(t/365)
    
    # Create meshgrid for plotting
    K, T = np.meshgrid(strikes, maturities)
    
    return K, T, vol_matrix

def update_vol_surface(df):
    """
    Updates volatility surface with new data
    
    Parameters:
    df: DataFrame with columns ['strike_price', 'days_to_expiry', 'implied_vol']
    """
    # Extract unique values
    strikes = sorted(df['strike_price'].unique())
    maturities = sorted(df['days_to_expiry'].unique())
    
    # Create grid
    K, T, vol = create_vol_grid(strikes, maturities, df['implied_vol'].values)
    return K, T, vol

def plot_vol_surface(K, T, vol):
    """Plot the volatility surface"""
    fig = go.Figure(data=[go.Surface(x=K, y=T/365, z=vol)])
    
    fig.update_layout(
        title='Real-time Implied Volatility Surface',
        scene = dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Maturity (Years)',
            zaxis_title='Implied Volatility',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=800
    )
    return fig

def generate_synthetic_data():
    """Generate synthetic option data similar to Polygon structure"""
    strikes = np.linspace(80, 120, 10)
    maturities = np.array([7, 14, 30, 60, 90])  # days to expiry
    
    data = []
    for strike in strikes:
        for maturity in maturities:
            data.append({
                'strike_price': strike,
                'days_to_expiry': maturity,
                'implied_vol': None  # Will be filled in grid creation
            })
    
    return pd.DataFrame(data)

def main():
    st.title('Real-time Option Volatility Surface')
    
    # Initialize session state for data
    if 'data' not in st.session_state:
        st.session_state.data = generate_synthetic_data()
    
    # Create and plot surface
    K, T, vol = update_vol_surface(st.session_state.data)
    fig = plot_vol_surface(K, T, vol)
    
    # Display the plot
    st.plotly_chart(fig)
    
    # Display data table
    st.subheader("Current Option Data")
    st.dataframe(st.session_state.data)

if __name__ == "__main__":
    main()