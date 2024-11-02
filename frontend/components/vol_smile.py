import plotly.graph_objects as go
from typing import Dict, Any

class VolatilitySmile:
    def __init__(self):
        self.figure = None

    def create_smile_plot(self, df, timestamp) -> go.Figure:
        """Create volatility smile plot for specific timestamp"""
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
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig

    def create_term_structure_plot(self, df, timestamp) -> go.Figure:
        """Create volatility term structure plot"""
        df_slice = df[df['timestamp'] == timestamp]
        atm_options = df_slice[abs(df_slice['log_strike'] - 1) < 0.05]
        
        term_structure = atm_options.groupby('days_to_expiry')['implied_volatility'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=term_structure.index,
            y=term_structure.values,
            mode='markers+lines',
            name='Term Structure',
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Volatility Term Structure at {timestamp}",
            xaxis_title="Days to Expiry",
            yaxis_title="ATM Implied Volatility",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig