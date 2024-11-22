import dash
from dash import html
from dash.dependencies import Input, Output
from .layouts.main_layout import create_layout
from .components.rolling_animation import VolatilitySurfaceAnimation
from .components.vol_surface import VolatilitySurface
from .components.vol_smile import VolatilitySmile
from backend.core.data.DataHandler import DataHandler

class VolatilityDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.vol_surface_animation = VolatilitySurfaceAnimation()
        self.vol_surface = VolatilitySurface()
        self.vol_smile = VolatilitySmile()
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        self.app.layout = create_layout(
            self.vol_surface_animation,
            self.vol_surface,
            self.vol_smile
        )
        
    def setup_callbacks(self):
        @self.app.callback(
            Output('volatility-surface', 'figure'),
            Input('surface-interpolation-dropdown', 'value')
        )
        def update_surface(interpolation_type):
            if interpolation_type == 'raw':
                return self.vol_surface.create_raw_surface(self.data)
            return self.vol_surface.create_interpolated_surface(self.data)
            
        @self.app.callback(
            [Output('smile-graph', 'figure'),
             Output('term-structure-graph', 'figure')],
            Input('timestamp-dropdown', 'value')
        )
        def update_smile_and_term(timestamp):
            return (
                self.vol_smile.create_smile_plot(self.data, timestamp),
                self.vol_smile.create_term_structure_plot(self.data, timestamp)
            )

    def run_server(self, data, debug=True):
        self.data = data
        self.app.run_server(debug=debug)

if __name__ == "__main__":
    dashboard = VolatilityDashboard()
    # Load your data here
    data = DataProcessor.load_data("path/to/your/data")
    dashboard.run_server(data)