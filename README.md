# VolaSurfer - Volatility Surface Analysis Tool

## Project Overview
VolaSurfer is a Python-based analytical tool designed for advanced options data processing, implied volatility surface analysis, and dynamic hedging. This tool integrates concepts like *vega bucketing* to help quants, risk managers, and traders efficiently group options, analyze volatility skews/smiles, and manage exposures across different market conditions.

## Key Features
- **Implied Volatility Surface Analysis**: Extracts and visualizes implied volatilities, analyzing the volatility smile and skew.
- **Vega Bucketing**: Implements Nassim Taleb's concept of vega bucketing for grouping options by strike and maturity, analyzing vega exposures.
- **Dynamic Hedging and Stress Testing**: Helps assess vega exposures and calibrate risk models for better risk management.
- **Data Normalization**: For improved scalability in distribution and volatility visualization.
- **Options Data Aggregation**: Gathers options data at the ATM strike level, providing insights into the implied volatility for risk management and trading strategies.

## Use Case Examples
1. **Options Data Processing**: Easily consolidate and process (real-time) data for options chains, as seen in `SampleConsolidatedTradeBarLog.txt`.
2. **Visualization of Implied Volatility Over Time**: Plot implied volatility with timestamps for efficient tracking of the volatility smile or skew trends.
3. **Strike-Based Grouping**: Group and analyze options by strikes and maturities, enabling targeted risk assessment and hedging across different timeframes.

## How to Use
1. **Data Aggregation and Processing**:
   - Utilize `VolaSurfer.py` to collect, analyse, and visualise your options data!
   - [Soon to support live data!]

2. **Vega Bucketing**:
   - Configure vega bucketing based on the desired strike/maturity groupings.
   - Adjust risk management settings to monitor aggregate exposures.

3. **Volatility Surface Visualization**:
   - Execute the Python script to plot implied volatilities over time in a visually intuitive way.

## Contributions
Contributions are welcomed! If you wish to extend the functionality, feel free to comment your thoughts or even fork the repository, make your changes, and submit a pull request!

