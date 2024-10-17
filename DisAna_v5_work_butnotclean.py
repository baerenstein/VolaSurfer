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

print(df.columns)
print(df.head())


# Assuming 'S' (underlying price) is not in the dataset, we'll use the mid-price of the option as a proxy
df['S'] = (df['bid_open'] + df['ask_open']) / 2
df['r'] = 0.02  # Assuming risk-free rate is 2% (you may need to adjust this)

# Calculate implied volatility for all available prices
price_columns = ['bid_open', 'bid_high', 'bid_low', 'bid_close', 'ask_open', 'ask_high', 'ask_low', 'ask_close']

for col in price_columns:
    iv_col = f'iv_{col}'
    df[iv_col] = df.apply(lambda row: implied_volatility(
        row[col], row['S'], row['strike'], row['T'], row['r'], row['option_type']
    ), axis=1)

# Aggregate by timestamp
iv_columns = ['iv_' + col for col in price_columns]

df_agg = df.groupby('timestamp')[iv_columns].agg(['mean', 'median', 'std', 'skew', 
                                                  lambda x: stats.kurtosis(x, fisher=False)])
df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
df_agg = df_agg.reset_index()

# Rename the kurtosis columns
df_agg = df_agg.rename(columns={col: col.replace('<lambda>', 'kurt') for col in df_agg.columns if '<lambda>' in col})

# Create a DataFrame for distribution analysis
df_dist = df_agg.drop('timestamp', axis=1)


def plot_distributions(df):
    # Summary statistics
    summary_stats = df.agg(['mean', 'median', 'std', 'skew', 'kurt']).T
    summary_stats.columns = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis']
    print(summary_stats)

    # Histogram grid
    plt.figure(figsize=(20, 15))
    sns.set(style="whitegrid")
    melted_df = df.melt(var_name='Statistic', value_name='Value')
    g = sns.FacetGrid(melted_df, col="Statistic", col_wrap=4, height=4, aspect=1.5)
    g.map(sns.histplot, "Value", kde=True)
    g.set_axis_labels("Value", "Count")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.show()

    # Cumulative Distribution
    plt.figure(figsize=(15, 10))
    for column in df.columns:
        sorted_data = np.sort(df[column])
        yvals = np.arange(len(sorted_data))/float(len(sorted_data))
        plt.plot(sorted_data, yvals, label=column)
    plt.title('Cumulative Distribution of Implied Volatility Statistics')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Joyplot
    plt.figure(figsize=(15, 10))
    joyplot(
        data=df,
        title="Implied Volatility Statistics Distributions",
        labels=df.columns,
        colormap=plt.cm.viridis,
        overlap=0.7
    )
    plt.xlabel("Value")
    plt.tight_layout()
    plt.show()

    # Heatmap of Distribution Differences
    def distribution_difference(x, y):
        return stats.ks_2samp(x, y).statistic

    diff_matrix = df.apply(lambda x: df.apply(lambda y: distribution_difference(x, y)))

    plt.figure(figsize=(15, 12))
    sns.heatmap(diff_matrix, annot=True, cmap='YlOrRd')
    plt.title('Heatmap of Distribution Differences (K-S Statistic)')
    plt.tight_layout()
    plt.show()

    # Q-Q Plots
    n_cols = 4
    n_rows = (len(df.columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for i, column in enumerate(df.columns):
        stats.probplot(df[column].dropna(), dist="norm", plot=axes[i])
        axes[i].set_title(f'Q-Q Plot: {column}')

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Run the distribution analysis
plot_distributions(df_dist)