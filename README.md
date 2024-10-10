#TODO
# VolaSurfer


This project generates and animates implied volatility surfaces for various option pricing models. It visualizes how implied volatility changes across different strike prices and maturities, incorporating effects like volatility smile and skew.

## Features

- Supports multiple volatility models:
  - Constant volatility
  - Deterministic volatility
  - Stochastic volatility
  - Jump-diffusion
  - Uncertain volatility
  - Stochastic volatility with mean reversion

- Incorporates volatility smile and skew effects
- Animates the evolution of volatility surfaces over time

## Mathematical Formulations

The core of this script is based on generating volatility surfaces using different models. Here are some key mathematical formulations:

1. Deterministic Volatility:
   σ(t) = σ_base + slope * t

2. Stochastic Volatility:
   σ(t) = σ_deterministic(t) + vol_of_vol * ε,  where ε ~ N(0,1)

3. Jump Diffusion:
   σ(t) = σ_stochastic(t) + J * I,  where J ~ N(μ_jump, σ_jump) and I ~ Bernoulli(p_jump)

4. Volatility Smile and Skew:
   σ(K,t) = σ_model(t) + smile_intensity * (ln(K/S))^2 + skew_intensity * ln(K/S)

Where:
- σ(t) is the volatility at time t
- K is the strike price
- S is the spot price
- t is time to maturity

## Requirements

- NumPy
- Python 3.x
- Pandas
- Matplotlib

## Customization

You can modify the `model_params` dictionary to adjust the behavior of each volatility model. The main function `animate_volatility_surfaces` allows you to specify the spot price, chosen model, and number of days for the animation.

## Output

The script generates an interactive 3D plot showing the evolution of the implied volatility surface over time. The x-axis represents strike prices, the y-axis represents maturities, and the z-axis represents implied volatility.
