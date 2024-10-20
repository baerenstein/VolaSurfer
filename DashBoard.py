import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

class Dashboard:
    
    def __init__(self):
        self.app = dash.Dash(__name__)

    def display(self, df):
        self.create_layout(df)
        self.app.run_server(debug=True)

    def create_layout(self, df):

        # Synthetic data for 3D surface plot
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        x, y = np.meshgrid(x, y)
        z = np.sin(np.sqrt(x**2 + y**2))


        unique_timestamps = df['timestamp'].unique().to_numpy()
        unique_timestamps.sort()

        df_call = df[df['option_type'] == 'Call']
        df_call['days_to_expiry'] = (df_call['maturity'] - df_call['timestamp']).dt.total_seconds() / (24 * 60 * 60)

        # 3D Surface plot
        surface_plot = go.Figure(data=[go.Surface(
            x=x,y=y,z=z,
            colorscale='rdbu')])
        surface_plot.update_layout(
            title="VolaSurfer",
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis"
            )
        )

        heatmap_3x4 = make_subplots(rows=1, cols=1, subplot_titles=["VolaHeat"])

        heatmap = go.Heatmap(
            x=df_call[df_call['timestamp'] == unique_timestamps[0]]['days_to_expiry'],
            y=df_call[df_call['timestamp'] == unique_timestamps[0]]['log_strike'],
            z=df_call[df_call['timestamp'] == unique_timestamps[0]]['implied_volatility'],
            colorscale='Viridis',
            colorbar=dict(title='Implied Volatility'),
        )

        heatmap_3x4.add_trace(heatmap)

        frames = []
        for timestamp in unique_timestamps:
            df_slice = df_call[df_call['timestamp'] == timestamp]
            frame = go.Frame(
                data=[go.Heatmap(
                    x=df_slice['days_to_expiry'],
                    y=df_slice['log_strike'],
                    z=df_slice['implied_volatility'],
                    colorscale='inferno',
                )],
                name=str(timestamp)
            )
            frames.append(frame)

        heatmap_3x4.frames = frames

        heatmap_3x4.update_layout(
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


        # Synthetic data for 5x30 heatmap
        heatmap_5x30_data = np.random.randint(0, 100, (5, 30))
        heatmap_5x30 = go.Figure(data=go.Heatmap(
            z=heatmap_5x30_data,
            colorscale='magma'
        ))
        heatmap_5x30.update_layout(title="VolaSmile over Time")

        # Generate synthetic distributions (more quadratic shaped)
        np.random.seed(42)
        x_vals = np.linspace(-5, 5, 500)

        # Quadratic shaped distributions for the ridgeline plot
        distributions = {
            'Category 1': np.random.normal(loc=-2, scale=0.5, size=1000),
            'Category 2': np.random.normal(loc=0, scale=1, size=1000),
            'Category 3': np.random.normal(loc=2, scale=1.5, size=1000),
            'Category 4': np.random.normal(loc=4, scale=0.75, size=1000),
            'Category 5': np.random.normal(loc=-3, scale=0.7, size=1000),
            'Category 6': np.random.normal(loc=3, scale=1.2, size=1000),
        }

        # Create the ridge plot with quadratic appearance (smoother and wider)
        ridge_plot = go.Figure()

        for i, (label, data) in enumerate(distributions.items()):
            # Generate KDE (Kernel Density Estimate) for each distribution
            hist, bin_edges = np.histogram(data, bins=100, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            # Offset the curves vertically
            ridge_plot.add_trace(go.Scatter(
                x=bin_centers, y=hist - i * 0.25,  # Adjust the offset to make it more quadratic
                mode='lines', name=label, fill='tonexty' if i > 0 else None,
                line=dict(width=2)
            ))

        # Update layout for the ridge plot
        ridge_plot.update_layout(
            title="VolaRidge",
            yaxis=dict(title='', zeroline=False, showline=False, showgrid=False),
            xaxis=dict(title='X Axis', showgrid=False),
            showlegend=True,
            template="simple_white"
        )

        # Layout of the app
        self.app.layout = html.Div(children=[
            html.H1(children='VolaSurfer'),

            html.Div(children='''
                An options implied volatility dashboard.
            '''),

            # First row: 3D Surface Plot and 3x4 Heatmap side by side
            html.Div([
                html.Div(dcc.Graph(id='VolaSurfer', figure=surface_plot), style={'width': '50%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id='VolaHeat', figure=heatmap_3x4), style={'width': '50%', 'display': 'inline-block'})
            ]),

            # Second row: 5x30 Heatmap
            html.Div(dcc.Graph(id='VolaSmile over Time', figure=heatmap_5x30), style={'width': '100%'}),

            # Third row: Ridge Plot
            html.Div(dcc.Graph(id='VolaRidge', figure=ridge_plot), style={'width': '100%'})
        ])


        # nice colorscales rdbu, icefire

