import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import numpy as np
import pandas as pd

class Dashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)

    def display(self, df):
        self.create_layout(df)
        self.app.run_server(debug=True)

    def create_layout(self, df):
        self.app.layout = html.Div(children=[
            html.H1(children='VolaSurfer'),
            html.Div(children='An options implied volatility dashboard.'),
            html.Div([
                html.Div(dcc.Graph(id='VolaSurfer', figure=self.create_surface_plot()), style={'width': '50%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id='VolaHeat', figure=self.create_heatmap(df)), style={'width': '50%', 'display': 'inline-block'})
                ]),
            html.Div(dcc.Graph(id='VolaSmile over Time', figure=self.create_smile_heatmap(df)), style={'width': '100%'}),
            html.Div(dcc.Graph(id='VolaRidge', figure=self.create_ridge_plot()), style={'width': '100%'})
        ])

    #TODO load real data
    def create_surface_plot(self):
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        x, y = np.meshgrid(x, y)
        z = np.sin(np.sqrt(x**2 + y**2))

        surface_plot = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='rdbu')])
        surface_plot.update_layout(
            title="VolaSurfer",
            scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis")
        )
        return surface_plot

    #TODO adjust to plot bins
    def create_heatmap(self, df):
        df_call = df[df['option_type'] == 'Call']
        df_call['days_to_expiry'] = (df_call['maturity'] - df_call['timestamp']).dt.total_seconds() / (24 * 60 * 60)
        unique_timestamps = df['timestamp'].unique().to_numpy()
        unique_timestamps.sort()

        heatmap_3x4 = make_subplots(rows=1, cols=1, subplot_titles=["VolaHeat"])

        heatmap = go.Heatmap(
            x=df_call[df_call['timestamp'] == unique_timestamps[0]]['days_to_expiry'],
            y=df_call[df_call['timestamp'] == unique_timestamps[0]]['log_strike'],
            z=df_call[df_call['timestamp'] == unique_timestamps[0]]['implied_volatility'],
            colorscale='inferno',
            colorbar=dict(title='Implied Volatility'),
        )

        heatmap_3x4.add_trace(heatmap)
        heatmap_3x4.frames = self.create_heatmap_frames(df_call, unique_timestamps)
        heatmap_3x4.update_layout(
            updatemenus=[self.create_play_button()],
            sliders=[self.create_timestamp_slider(heatmap_3x4.frames)]
        )
        return heatmap_3x4

    def create_heatmap_frames(self, df_call, unique_timestamps):
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
        return frames

    def create_play_button(self):
        return dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]
            )]
        )

    def create_timestamp_slider(self, frames):
        return dict(
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
        )

    #TODO load data properly and visualise on evolution of data point basis
    def create_smile_heatmap(self, df):
        # Ensure the DataFrame is sorted by timestamp and maturity
        df = df.sort_values(['timestamp', 'maturity', 'strike'])
        
        # Get unique timestamps
        timestamps = df['timestamp'].unique()
        
        # Initialize the figure
        fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2],
                            shared_xaxes=True, vertical_spacing=0.02)
        
        # Initialize data storage for animation
        smile_data = []
        atm_data = []

        # Create a color scale for the bar chart
        colorscale = pc.sequential.Viridis

        frames = []
        for i, timestamp in enumerate(timestamps):
            # Filter data for the current timestamp
            df_t = df[df['timestamp'] == timestamp]
            
            # Get the closest expiry
            closest_expiry = df_t['maturity'].min()
            df_closest = df_t[df_t['maturity'] == closest_expiry]
            
            # Sort by strike price
            df_closest = df_closest.sort_values('strike')
            
            # Append data for the smile heatmap
            smile_data.append(df_closest['implied_volatility'].values)
            
            # Get at-the-money volatility (assume it's the middle strike)
            atm_vol = df_closest.iloc[len(df_closest)//2]['implied_volatility']
            atm_data.append(atm_vol)
            
            # Create frame
            frame = go.Frame(
                data=[
                    go.Heatmap(z=[smile_data], x=df_closest['strike'], y=list(range(len(smile_data))),
                            colorscale='Viridis', showscale=False),
                    go.Bar(x=timestamps[:i+1], y=atm_data, 
                        marker=dict(color=atm_data, colorscale=colorscale))
                ],
                name=str(timestamp)
            )
            frames.append(frame)
        
        # Create the main figure
        fig = go.Figure(
            data=[
                go.Heatmap(z=[smile_data[0]], x=df[df['timestamp'] == timestamps[0]]['strike'], y=[0],
                        colorscale='Viridis', showscale=False),
                go.Bar(x=[timestamps[0]], y=[atm_data[0]], 
                    marker=dict(color=[atm_data[0]], colorscale=colorscale))
            ],
            frames=frames
        )
        
        # Update layout
        fig.update_layout(
            title="Volatility Smile Evolution and ATM Volatility",
            updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
                            method='animate', args=[None, dict(frame=dict(duration=300, redraw=True),
                            fromcurrent=True, mode='immediate')])])],
            xaxis=dict(title='Strike Price'),
            yaxis=dict(title='Time', showticklabels=False),
            xaxis2=dict(title='Timestamp'),
            yaxis2=dict(title='ATM Volatility')
        )
        
        # Add slider
        fig.update_layout(
            sliders=[dict(
                steps=[dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=300, redraw=True))], label=f.name)
                    for f in frames],
                transition=dict(duration=0),
                x=0, y=0, 
                currentvalue=dict(font=dict(size=12), prefix='Timestamp: ', visible=True, xanchor='center'),
                len=1.0
            )]
        )
        
        return fig

    #TODO load real data
    def create_ridge_plot(self):
        np.random.seed(42)
        x_vals = np.linspace(-5, 5, 500)
        distributions = self.generate_distributions()

        ridge_plot = go.Figure()

        for i, (label, data) in enumerate(distributions.items()):
            hist, bin_edges = np.histogram(data, bins=100, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            ridge_plot.add_trace(go.Scatter(
                x=bin_centers, y=hist - i * 0.25,
                mode='lines', name=label, fill='tonexty' if i > 0 else None,
                line=dict(width=2)
            ))

        ridge_plot.update_layout(
            title="VolaRidge",
            yaxis=dict(title='', zeroline=False, showline=False, showgrid=False),
            xaxis=dict(title='X Axis', showgrid=False),
            showlegend=True,
            template="simple_white"
        )
        return ridge_plot

    #TODO delete once obsolete
    def generate_distributions(self):
        return {
            'Category 1': np.random.normal(loc=-2, scale=0.5, size=1000),
            'Category 2': np.random.normal(loc=0, scale=1, size=1000),
            'Category 3': np.random.normal(loc=2, scale=1.5, size=1000),
            'Category 4': np.random.normal(loc=4, scale=0.75, size=1000),
            'Category 5': np.random.normal(loc=-3, scale=0.7, size=1000),
            'Category 6': np.random.normal(loc=3, scale=1.2, size=1000),
        }