import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from joypy import joyplot
from scipy.optimize import brentq

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def implied_volatility(price, S, K, T, r, option_type):
    def objective(sigma):
        if option_type == 'Call':
            return black_scholes_call(S, K, T, r, sigma) - price
        else:
            return black_scholes_put(S, K, T, r, sigma) - price
    try:
        return brentq(objective, 1e-9, 10)
    except ValueError:
        return np.nan

# Load and preprocess the data
df = pd.read_excel('options_data.xlsx', parse_dates=['timestamp'])
df['S'] = 168.0  # Assuming underlying price is 168 (you may need to adjust this)
df['r'] = 0.02  # Assuming risk-free rate is 2% (you may need to adjust this)

# Calculate implied volatility for bid_open
df['iv_bid_open'] = df.apply(lambda row: implied_volatility(
    row['bid_open'], row['S'], row['strike'], row['T'], row['r'], row['option_type']
), axis=1)

# Aggregate by timestamp
df_agg = df.groupby('timestamp')['iv_bid_open'].agg([
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('skew', 'skew'),
    ('kurt', lambda x: stats.kurtosis(x))
]).reset_index()

# Create a DataFrame for distribution analysis
contracts = ['mean', 'median', 'std', 'skew', 'kurt']
data = {contract: df_agg[contract] for contract in contracts}
df_dist = pd.DataFrame(data)

# Distribution analysis
def plot_distributions(df):
    # Summary statistics
    summary_stats = df.agg(['mean', 'median', 'std', 'skew', 'kurt']).T
    summary_stats.columns = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis']
    print(summary_stats)

    # Histogram grid
    plt.figure(figsize=(16, 12))
    sns.set(style="whitegrid")
    melted_df = df.melt(var_name='Statistic', value_name='Value')
    g = sns.FacetGrid(melted_df, col="Statistic", col_wrap=2, height=4, aspect=1.5)
    g.map(sns.histplot, "Value", kde=True)
    g.set_axis_labels("Value", "Count")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.show()

    # Cumulative Distribution
    plt.figure(figsize=(12, 6))
    for contract in df.columns:
        sorted_data = np.sort(df[contract])
        yvals = np.arange(len(sorted_data))/float(len(sorted_data))
        plt.plot(sorted_data, yvals, label=contract)
    plt.title('Cumulative Distribution of Implied Volatility Statistics')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Joyplot
    plt.figure(figsize=(12, 8))
    joyplot(
        data=df,
        title="Implied Volatility Statistics Distributions",
        labels=df.columns,
        colormap=plt.cm.viridis,
        overlap=0.7
    )
    plt.xlabel("Value")
    plt.show()

    # Heatmap of Distribution Differences
    def distribution_difference(x, y):
        return stats.ks_2samp(x, y).statistic

    diff_matrix = df.apply(lambda x: df.apply(lambda y: distribution_difference(x, y)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_matrix, annot=True, cmap='YlOrRd')
    plt.title('Heatmap of Distribution Differences (K-S Statistic)')
    plt.show()

    # Q-Q Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, contract in enumerate(df.columns):
        stats.probplot(df[contract], dist="norm", plot=axes[i])
        axes[i].set_title(f'Q-Q Plot: {contract}')

    plt.tight_layout()
    plt.show()

# Run the distribution analysis
plot_distributions(df_dist)