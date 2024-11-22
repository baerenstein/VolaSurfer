import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime
import asyncio

from .api_client import VolatilityAPIClient
from .components.surface import create_surface_plot
from .components.smile import create_smile_plot
from .components.heatmap import create_heatmap_plot
from .layouts.main_layout import create_main_layout

class VolatilityDashboard:
    """A dashboard for visualizing and analyzing option volatility surfaces"""
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.logger = logging.getLogger(__name__)
        self.api_client = VolatilityAPIClient()
        self.setup_logging()

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def initialize(self):
        """Initialize the dashboard with API health check"""
        try:
            health = await self.api_client.get_health()
            if not health.get("data_loaded"):
                raise Exception("API data not loaded")
            self.logger.info("Successfully connected to API")
        except Exception as e:
            self.logger.error(f"Failed to initialize API connection: {str(e)}")
            raise

    def create_callbacks(self):
        """Set up all dashboard callbacks"""
        @self.app.callback(
            Output('timestamp-dropdown', 'options'),
            Input('interval-component', 'n_intervals')
        )
        async def update_timestamps(_):
            try:
                data = await self.api_client.get_timestamps()
                timestamps = data["timestamps"]
                return [{'label': ts, 'value': ts} for ts in timestamps]
            except Exception as e:
                self.logger.error(f"Error updating timestamps: {str(e)}")
                return []

        @self.app.callback(
            Output('volatility-surface', 'figure'),
            [Input('surface-interpolation-dropdown', 'value'),
             Input('timestamp-dropdown', 'value')]
        )
        async def update_surface(interpolation, timestamp):
            try:
                data = await self.api_client.get_volatility_surface(
                    timestamp=timestamp,
                    interpolated=(interpolation == 'interpolated')
                )
                return create_surface_plot(data)
            except Exception as e:
                self.logger.error(f"Error updating surface: {str(e)}")
                return go.Figure()

        @self.app.callback(
            [Output('smile-graph', 'figure'),
             Output('term-structure-graph', 'figure')],
            Input('timestamp-dropdown', 'value')
        )
        async def update_graphs(timestamp):
            try:
                smile_data = await self.api_client.get_volatility_smile(timestamp)
                return create_smile_plot(smile_data), create_term_structure_plot(smile_data)
            except Exception as e:
                self.logger.error(f"Error updating graphs: {str(e)}")
                return go.Figure(), go.Figure()

    async def run_server(self, debug=True):
        """Run the dashboard server"""
        try:
            await self.initialize()
            self.app.layout = create_main_layout()
            self.create_callbacks()
            self.app.run_server(debug=debug)
        except Exception as e:
            self.logger.error(f"Error running server: {str(e)}")
        finally:
            await self.api_client.close()