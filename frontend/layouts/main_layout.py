from dash import html, dcc
from dash.dependencies import Input, Output

def create_layout(vol_surface_animation, vol_surface, vol_smile):
    return html.Div([
        html.H1('Options Volatility Analysis Dashboard', 
               style={'textAlign': 'center', 'padding': '20px'}),
        
        dcc.Tabs([
            dcc.Tab(label='Surface and Heatmap', children=[
                html.Div([
                    # Volatility Surface Section
                    html.Div([
                        dcc.Graph(id='volatility-surface', 
                                style={'width': '70%', 'display': 'inline-block'}),
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
                            html.P("The volatility surface shows how implied volatility varies with both strike price and time to expiration.")
                        ], style={'width': '25%', 'float': 'right', 'padding': '20px'})
                    ]),
                    
                    # Rolling Animation Section
                    html.Div([
                        dcc.Graph(
                            figure=vol_surface_animation.create_figure(),
                            style={'width': '70%', 'display': 'inline-block'}
                        ),
                        html.Div([
                            html.H3("Rolling Volatility Animation"),
                            html.P("This animation shows how the volatility surface evolves over time using a moving window approach.")
                        ], style={'width': '25%', 'float': 'right', 'padding': '20px'})
                    ])
                ])
            ]),
            
            dcc.Tab(label='Smile and Term Structure', children=[
                html.Div([
                    # Timestamp selector
                    dcc.Dropdown(
                        id='timestamp-dropdown',
                        style={'width': '50%', 'margin': '10px auto'}
                    ),
                    
                    # Smile and Term Structure plots
                    html.Div([
                        html.Div([
                            dcc.Graph(id='smile-graph', style={'width': '100%'}),
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Graph(id='term-structure-graph', style={'width': '100%'}),
                        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                    ]),
                    
                    # Explanations
                    html.Div([
                        html.Div([
                            html.H3("Volatility Smile Explanation"),
                            html.P("The volatility smile shows how implied volatility varies with strike price for a given expiration date.")
                        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
                        html.Div([
                            html.H3("Term Structure Explanation"),
                            html.P("The volatility term structure shows how implied volatility varies with time to expiration for at-the-money options.")
                        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'vertical-align': 'top'})
                    ])
                ])
            ])
        ])
    ])