import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def constant_volatility(strikes, maturities, spot, params):
    return np.full((len(maturities), len(strikes)), params['constant_vol'])

def deterministic_volatility(strikes, maturities, spot, params):
    return params['base_vol'] + params['slope'] * maturities[:, np.newaxis]

def stochastic_volatility(strikes, maturities, spot, params):
    base_vol = deterministic_volatility(strikes, maturities, spot, params)
    return base_vol + params['vol_of_vol'] * np.random.randn(len(maturities), len(strikes))

def jump_diffusion(strikes, maturities, spot, params):
    base_vol = stochastic_volatility(strikes, maturities, spot, params)
    jump_probability = np.random.rand(len(maturities), len(strikes)) < params['jump_prob']
    jump_size = np.random.normal(params['jump_mean'], params['jump_std'], (len(maturities), len(strikes)))
    return base_vol + jump_probability * jump_size

def uncertain_volatility(strikes, maturities, spot, params):
    base_vol = deterministic_volatility(strikes, maturities, spot, params)
    uncertainty = params['uncertainty'] * np.random.rand(len(maturities), len(strikes))
    return np.clip(base_vol + uncertainty, params['min_vol'], params['max_vol'])

def stoch_vol_mean_variance(strikes, maturities, spot, params):
    base_vol = stochastic_volatility(strikes, maturities, spot, params)
    mean_reversion = params['mean_reversion'] * (params['long_term_mean'] - base_vol)
    return base_vol + mean_reversion * maturities[:, np.newaxis]

def add_smile_and_skew(volatility_surface, strikes, spot, params): 
    #TODO
    moneyness = np.log(strikes / spot)
    smile = params.get('smile_intensity', 0) * (moneyness**2)
    skew = params.get('skew_intensity', 0) * moneyness
    return volatility_surface + smile + skew

def volatility_surface(strikes, maturities, spot, model='stochastic', params=None, time_step=0):
    if params is None:
        params = {}
    
    model_functions = {
        'constant': constant_volatility,
        'deterministic': deterministic_volatility,
        'stochastic': stochastic_volatility,
        'jump_diffusion': jump_diffusion,
        'uncertain': uncertain_volatility,
        'stoch_mean_var': stoch_vol_mean_variance
    }
    
    if model not in model_functions:
        raise ValueError(f"Unknown model: {model}")
    
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    
    # Generate base volatility surface using the selected model
    iv_surface = model_functions[model](strikes, maturities, spot, params)
    
    # Add smile and skew effect
    iv_surface = add_smile_and_skew(iv_surface, strikes, spot, params)
    
    # Add time-varying component
    time_variation = params.get('time_variation', 0) * 0.02 * np.sin(time_step * 0.1 + maturity_grid * np.pi)
    iv_surface += time_variation
    
    # Ensure volatility is within reasonable bounds
    iv_surface = np.clip(iv_surface, 0.05, 0.5)
    
    return strike_grid, maturity_grid, iv_surface

def update_surface(frame, ax, spot, model, params):
    ax.clear()
    strikes = np.linspace(spot * 0.5, spot * 1.5, 50)  # Wider range of strikes
    maturities = np.linspace(1/12, 2, 50)  # Maturities from 1 month to 2 years
    
    strike_grid, maturity_grid, iv_surface = volatility_surface(strikes, maturities, spot, model, params, time_step=frame)
    
    surf = ax.plot_surface(strike_grid, maturity_grid, iv_surface, cmap='viridis', 
                           edgecolor='none', alpha=0.8)
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity (years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Implied Volatility Surface')
    # ax.set_title(f'Implied Volatility Surface\n(Model: {model})')
    ax.set_zlim(0, 0.5)  # Adjust if needed to see the full range of volatilities
    ax.view_init(elev=90, azim=0)   # Set the viewing angle
    
    return surf,

def animate_volatility_surfaces(spot, model='stochastic', params=None, num_days=30):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    anim = FuncAnimation(fig, update_surface, frames=num_days, 
                         fargs=(ax, spot, model, params),
                         interval=500, blit=False, repeat=True)
    
    plt.show()

# Example usage
spot_price = 1.0  

# Parameters for different models with enhanced smile and skew
model_params = {
    'constant': {'constant_vol': 0.15, 'smile_intensity': 0.1, 'skew_intensity': 0.05},
    'deterministic': {'base_vol': 0.1, 'slope': 0.05, 'smile_intensity': 0.1, 'skew_intensity': 0.05},
    'stochastic': {'base_vol': 0.1, 'slope': 0.05, 'vol_of_vol': 0.02, 'smile_intensity': 0.1, 'skew_intensity': 0.05},
    'jump_diffusion': {'base_vol': 0.1, 'slope': 0.05, 'vol_of_vol': 0.02, 'jump_prob': 0.01, 'jump_mean': 0, 'jump_std': 0.05, 'smile_intensity': 0.1, 'skew_intensity': 0.05},
    'uncertain': {'base_vol': 0.1, 'slope': 0.05, 'uncertainty': 0.05, 'min_vol': 0.05, 'max_vol': 0.3, 'smile_intensity': 0.1, 'skew_intensity': 0.05},
    'stoch_mean_var': {'base_vol': 0.1, 'slope': 0.05, 'vol_of_vol': 0.02, 'mean_reversion': 0.5, 'long_term_mean': 0.15, 'smile_intensity': 0.1, 'skew_intensity': 0.05}
}

# Choose a model to animate
chosen_model = 'jump_diffusion'
animate_volatility_surfaces(spot_price, model=chosen_model, params=model_params[chosen_model])