import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy.interpolate import griddata

from DataHandler import DataHandler

class VolatilityDashboard:
    """A dashboard for visualizing and analyzing option volatility surfaces and related metrics"""
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.logger = logging.getLogger(__name__)
        self.data_handler = DataHandler()
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def prepare_data(self, file_path):
        """Prepare options data using DataHandler"""
        try:
            df = self.data_handler.parse_file(file_path)
            df = self.data_handler.process_data(df)
            df['days_to_expiry'] = (df['maturity'] - df['timestamp']).dt.total_seconds() / (24 * 60 * 60)
            df = df[df['implied_volatility'].notna() & 
                   (df['implied_volatility'] > 0) & 
                   (df['implied_volatility'] < 5)]
            self.logger.info(f"Successfully prepared data with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def interpolate_volatility_surface(self, df):
        """Interpolate the volatility surface"""
        df_calls = df[df['option_type'] == 'Call']
        
        # Create a grid for interpolation
        grid_x, grid_y = np.mgrid[df_calls['days_to_expiry'].min():df_calls['days_to_expiry'].max():100j, 
                                  df_calls['log_strike'].min():df_calls['log_strike'].max():100j]
        
        # Interpolate
        grid_z = griddata((df_calls['days_to_expiry'], df_calls['log_strike']), 
                          df_calls['implied_volatility'], 
                          (grid_x, grid_y), 
                          method='cubic')
        
        return grid_x, grid_y, grid_z

    def create_volatility_surface(self, df, interpolate=False):
        """Create 3D volatility surface visualization"""
        try:
            if interpolate:
                grid_x, grid_y, grid_z = self.interpolate_volatility_surface(df)
                
                surface = go.Figure(data=[go.Surface(
                    z=grid_z,
                    x=grid_x[0],
                    y=grid_y[:,0],
                    colorscale='rdbu',
                    colorbar=dict(title='Implied Volatility', titleside='right')
                )])
                title = "Interpolated Implied Volatility Surface"
            else:
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
                title = "Raw Implied Volatility Surface"

            surface.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Days to Expiry",
                    yaxis_title="Log Strike",
                    zaxis_title="Implied Volatility",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                margin=dict(l=65, r=50, b=65, t=90)
            )
            return surface
        except Exception as e:
            self.logger.error(f"Error creating volatility surface: {str(e)}")
            return go.Figure()

    def create_smile_plot(self, df, timestamp):
        """Create volatility smile plot for specific timestamp"""
        try:
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
                showlegend=True
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating smile plot: {str(e)}")
            return go.Figure()

    def create_term_structure_plot(self, df, timestamp):
        """Create volatility term structure plot"""
        try:
            df_slice = df[df['timestamp'] == timestamp]
            atm_options = df_slice[abs(df_slice['log_strike'] - 1) < 0.05]
            term_structure = atm_options.groupby('days_to_expiry')['implied_volatility'].mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=term_structure.index,
                y=term_structure.values,
                mode='markers+lines',
                name='ATM Term Structure'
            ))
            fig.update_layout(
                title=f"ATM Volatility Term Structure at {timestamp}",
                xaxis_title="Days to Expiry",
                yaxis_title="ATM Implied Volatility"
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating term structure plot: {str(e)}")
            return go.Figure()

    def create_volatility_heatmap(self, df):
        """Create volatility heatmap using binned implied volatilities"""
        try:
            df['log_strike_bin'] = pd.cut(df['log_strike'], bins=20)
            df['days_to_expiry_bin'] = pd.cut(df['days_to_expiry'], bins=20)
            pivot = df.pivot_table(
                values='implied_volatility',
                index='log_strike_bin',
                columns='days_to_expiry_bin',
                aggfunc='mean'
            )
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                colorscale='Viridis',
                colorbar=dict(title='Implied Volatility')
            ))
            fig.update_layout(
                title="Implied Volatility Heatmap",
                xaxis_title="Days to Expiry",
                yaxis_title="Log Strike",
                xaxis_tickangle=-45
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating volatility heatmap: {str(e)}")
            return go.Figure()

    def create_layout(self, df):
        """Create the dashboard layout"""
        self.app.layout = html.Div([
            html.H1('Options Volatility Analysis Dashboard', 
                   style={'textAlign': 'center', 'padding': '20px'}),
            
            dcc.Tabs([
                dcc.Tab(label='Surface and Heatmap', children=[
                    html.Div([
                        html.Div([
                            dcc.Graph(id='volatility-surface', style={'width': '70%', 'display': 'inline-block'}),
                            html.Div([
                                dcc.Dropdown(
                                    id='surface-interpolation-dropdown',
                                    options=[
                                        {'label': 'Raw Surface', 'value': 'raw'},
                                        {'label': 'Interpolated Surface', 'value': 'interpolated'}
                                    ],
                                    value='raw',
                                    style={'width': '100%', 'margin': '10px 0'}
                                ),
                                html.H3("Volatility Surface Explanation"),
                                html.P("The volatility surface shows how implied volatility varies with both strike price and time to expiration. The x-axis represents days to expiry, the y-axis represents the log strike price, and the z-axis (color) represents the implied volatility.")
                            ], style={'width': '25%', 'float': 'right', 'padding': '20px'})
                        ]),
                        html.Div([
                            dcc.Graph(figure=self.create_volatility_heatmap(df), style={'width': '70%', 'display': 'inline-block'}),
                            html.Div([
                                html.H3("Volatility Heatmap Explanation"),
                                html.P("The heatmap provides a 2D view of the volatility surface. Darker colors indicate higher implied volatility. This visualization helps identify patterns in volatility across different strike prices and expiration dates.")
                            ], style={'width': '25%', 'float': 'right', 'padding': '20px'})
                        ])
                    ])
                ]),
                
                dcc.Tab(label='Smile and Term Structure', children=[
                    html.Div([
                        dcc.Dropdown(
                            id='timestamp-dropdown',
                            options=[{'label': str(ts), 'value': ts} for ts in sorted(df['timestamp'].unique())],
                            value=sorted(df['timestamp'].unique())[0],
                            style={'width': '50%', 'margin': '10px auto'}
                        ),
                        html.Div([
                            html.Div([
                                dcc.Graph(id='smile-graph', style={'width': '100%'}),
                            ], style={'width': '48%', 'display': 'inline-block'}),
                            html.Div([
                                dcc.Graph(id='term-structure-graph', style={'width': '100%'}),
                            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                        ]),
                        html.Div([
                            html.Div([
                                html.H3("Volatility Smile Explanation"),
                                html.P("The volatility smile shows how implied volatility varies with strike price for a given expiration date. The 'smile' shape often observed is due to higher implied volatilities for out-of-the-money options.")
                            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
                            html.Div([
                                html.H3("Term Structure Explanation"),
                                html.P("The volatility term structure shows how implied volatility varies with time to expiration for at-the-money options. This helps visualize the market's expectation of future volatility over different time horizons.")
                            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'vertical-align': 'top'})
                        ])
                    ])
                ])
            ])
        ])

        @self.app.callback(
            Output('volatility-surface', 'figure'),
            Input('surface-interpolation-dropdown', 'value')
        )
        def update_volatility_surface(interpolation):
            return self.create_volatility_surface(df, interpolate=(interpolation == 'interpolated'))

        @self.app.callback(
            [Output('smile-graph', 'figure'),
             Output('term-structure-graph', 'figure')],
            Input('timestamp-dropdown', 'value')
        )
        def update_graphs(timestamp):
            return self.create_smile_plot(df, timestamp), self.create_term_structure_plot(df, timestamp)

    def run_server(self, file_path, debug=True):
        """Run the dashboard server"""
        try:
            df = self.prepare_data(file_path)
            self.create_layout(df)
            self.app.run_server(debug=debug)
        except Exception as e:
            self.logger.error(f"Error running server: {str(e)}")

# Create dashboard instance
dashboard = VolatilityDashboard()

# Run dashboard with data file
dashboard.run_server("Data/SPY_Options_log.txt")