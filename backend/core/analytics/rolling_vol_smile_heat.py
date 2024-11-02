import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

class VolatilitySurfaceAnimation:
    def __init__(self):
        # Parameters
        self.maturities = 10
        self.strikes = 10
        self.total_columns = 50
        self.interval = 500  # Animation interval in ms

        # Generate synthetic data
        np.random.seed(42)
        self.data = np.abs(np.random.normal(loc=0.2, scale=0.05, 
                                          size=(self.maturities, self.total_columns)))
        
        # Initialize window
        self.window = np.full((self.maturities, self.strikes), np.nan)

    def create_frames(self):
        frames = []
        window = np.full((self.maturities, self.strikes), np.nan)

        for i in range(self.total_columns):
            if i < self.strikes:
                # Fill window from left to right
                window[:, i] = self.data[:, i]
            else:
                # Shift window left and add new column
                window[:, :-1] = window[:, 1:]
                window[:, -1] = self.data[:, i]

            frame = go.Frame(
                data=[go.Heatmap(
                    z=window,
                    colorscale='Viridis',
                    zmin=0.1,
                    zmax=0.6,
                    colorbar=dict(title='Implied Volatility')
                )],
                name=f'frame{i}'
            )
            frames.append(frame)
        
        return frames

    def create_figure(self):
        # Create initial heatmap
        initial_heatmap = go.Heatmap(
            z=np.zeros((self.maturities, self.strikes)),
            colorscale='Viridis',
            zmin=0.1,
            zmax=0.6,
            colorbar=dict(title='Implied Volatility')
        )

        # Create figure
        fig = go.Figure(
            data=[initial_heatmap],
            frames=self.create_frames(),
            layout=go.Layout(
                title=dict(
                    text='Dynamic Volatility Smile with Moving Window',
                    x=0.5,
                    font=dict(size=18)
                ),
                xaxis=dict(
                    title='Strikes',
                    tickmode='linear',
                    tick0=0,
                    dtick=1
                ),
                yaxis=dict(
                    title='Maturities (days to expiry)',
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    autorange='reversed'
                ),
                updatemenus=[dict(
                    type='buttons',
                    showactive=False,
                    x=0.1,
                    y=-0.15,
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(
                                mode='immediate',
                                fromcurrent=True,
                                frame=dict(duration=self.interval, redraw=True),
                                transition=dict(duration=0)
                            )]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[[None], dict(
                                mode='immediate',
                                transition=dict(duration=0),
                                frame=dict(duration=0, redraw=True)
                            )]
                        )
                    ]
                )]
            )
        )

        return fig

def main():
    # Create and display the animation
    vol_surface = VolatilitySurfaceAnimation()
    fig = vol_surface.create_figure()
    
    # Show the figure
    fig.show()

if __name__ == "__main__":
    main()

    