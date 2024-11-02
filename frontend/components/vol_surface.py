import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any

class VolatilitySurface:
    def __init__(self):
        self.figure = None

    def create_raw_surface(self, df) -> go.Figure:
        """Create 3D scatter plot of raw volatility surface"""
        df_calls = df[df['option_type'] == 'Call']
        
        surface = go.Figure(data=[go.Scatter3d(
            x=df_calls['days_to_expiry'],
            y=df_calls['log_strike'],
            z=df_calls['implied_volatility'],
            mode='markers',
            marker=dict(
                size=4,
                color=df_calls['implied_volatility'],
                colorscale='rdbu',
                opacity=0.8
            )
        )])

        surface.update_layout(
            title="Raw Implied Volatility Surface",
            scene=dict(
                xaxis_title="Days to Expiry",
                yaxis_title="Log Strike",
                zaxis_title="Implied Volatility",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=65, r=50, b=65, t=90)
        )
        
        return surface

    def create_interpolated_surface(self, df) -> go.Figure:
        """Create interpolated volatility surface"""
        df_calls = df[df['option_type'] == 'Call']
        
        # Create a grid for interpolation
        grid_x, grid_y = np.mgrid[
            df_calls['days_to_expiry'].min():df_calls['days_to_expiry'].max():100j,
            df_calls['log_strike'].min():df_calls['log_strike'].max():100j
        ]
        
        # Interpolate
        from scipy.interpolate import griddata
        grid_z = griddata(
            (df_calls['days_to_expiry'], df_calls['log_strike']),
            df_calls['implied_volatility'],
            (grid_x, grid_y),
            method='cubic'
        )
        
        surface = go.Figure(data=[go.Surface(
            z=grid_z,
            x=grid_x[0],
            y=grid_y[:,0],
            colorscale='rdbu',
            colorbar=dict(title='Implied Volatility', titleside='right')
        )])

        surface.update_layout(
            title="Interpolated Implied Volatility Surface",
            scene=dict(
                xaxis_title="Days to Expiry",
                yaxis_title="Log Strike",
                zaxis_title="Implied Volatility",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=65, r=50, b=65, t=90)
        )
        
        return surface
