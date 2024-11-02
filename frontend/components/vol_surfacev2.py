import plotly.graph_objects as go

def create_surface_plot(data: dict) -> go.Figure:
    """Create volatility surface plot"""
    fig = go.Figure(data=[
        go.Surface(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            colorscale='Viridis'
        )
    ])
    
    fig.update_layout(
        title='Implied Volatility Surface',
        scene={
            'xaxis_title': 'Days to Expiry',
            'yaxis_title': 'Log Strike',
            'zaxis_title': 'Implied Volatility'
        }
    )
    
    return fig