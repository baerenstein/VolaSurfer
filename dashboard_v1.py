import dash
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Synthetic data for 3D surface plot
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# 3D Surface plot
surface_plot = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
surface_plot.update_layout(
    title="Synthetic 3D Surface Plot",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

# Synthetic data for 3x4 heatmap
heatmap_3x4 = go.Figure(data=go.Heatmap(
    z=[[1, 20, 30, 40],
       [50, 60, 70, 80],
       [90, 100, 110, 120]],
    x=['A', 'B', 'C', 'D'],
    y=['X', 'Y', 'Z'],
    colorscale='Viridis'
))
heatmap_3x4.update_layout(title="3x4 Heatmap")

# Synthetic data for 5x30 heatmap
heatmap_5x30_data = np.random.randint(0, 100, (5, 30))
heatmap_5x30 = go.Figure(data=go.Heatmap(
    z=heatmap_5x30_data,
    colorscale='Cividis'
))
heatmap_5x30.update_layout(title="5x30 Heatmap")

# Create ridge plot data (simulate using line plots)
np.random.seed(0)
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
x_vals = np.linspace(0, 10, 500)
ridge_plot = go.Figure()

for i, cat in enumerate(categories):
    y_vals = np.sin(x_vals + i) * (1 + 0.5 * np.random.rand(len(x_vals)))
    ridge_plot.add_trace(go.Scatter(
        x=x_vals, y=y_vals - i * 2, mode='lines',
        name=cat, fill='tonexty' if i > 0 else None
    ))

ridge_plot.update_layout(
    title="Ridge Plot",
    yaxis=dict(title='Category', zeroline=False, showline=False, showgrid=False),
    xaxis=dict(title='X Axis'),
    showlegend=False
)

# Layout of the app
app.layout = html.Div(children=[
    html.H1(children='Plotly Dash Dashboard'),

    html.Div(children='''
        A dashboard displaying a 3D surface plot, two heatmaps, and a ridge plot.
    '''),

    # First row: 3D Surface Plot and 3x4 Heatmap side by side
    html.Div([
        html.Div(dcc.Graph(id='surface-plot', figure=surface_plot), style={'width': '50%', 'display': 'inline-block'}),
        html.Div(dcc.Graph(id='heatmap-3x4', figure=heatmap_3x4), style={'width': '50%', 'display': 'inline-block'})
    ]),

    # Second row: 5x30 Heatmap
    html.Div(dcc.Graph(id='heatmap-5x30', figure=heatmap_5x30), style={'width': '100%'}),

    # Third row: Ridge Plot
    html.Div(dcc.Graph(id='ridge-plot', figure=ridge_plot), style={'width': '100%'})
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

# open via browser with http://127.0.0.1:8050/