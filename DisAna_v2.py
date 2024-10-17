import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)  # for reproducibility
contracts = ['Contract_A', 'Contract_B', 'Contract_C', 'Contract_D']
data = {contract: pd.Series(np.random.lognormal(mean=0.1, sigma=0.2, size=1000)) 
        for contract in contracts}
df = pd.DataFrame(data)

summary_stats = df.agg(['mean', 'median', 'std', 'skew', 'kurt']).T
summary_stats.columns = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis']
print(summary_stats)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title('Implied Volatility Distribution Across Contracts')
plt.ylabel('Implied Volatility')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=df)
plt.title('Implied Volatility Distribution Across Contracts')
plt.ylabel('Implied Volatility')
plt.show()

plt.figure(figsize=(12, 6))
for contract in contracts:
    sns.kdeplot(df[contract], label=contract)
plt.title('Implied Volatility Density Across Contracts')
plt.xlabel('Implied Volatility')
plt.ylabel('Density')
plt.legend()
plt.show()

from scipy import stats

def ks_test_matrix(df):
    n = len(df.columns)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            statistic, p_value = stats.ks_2samp(df.iloc[:, i], df.iloc[:, j])
            matrix[i, j] = matrix[j, i] = p_value
    return pd.DataFrame(matrix, index=df.columns, columns=df.columns)

ks_matrix = ks_test_matrix(df)
print("Kolmogorov-Smirnov Test p-values:")
print(ks_matrix)

