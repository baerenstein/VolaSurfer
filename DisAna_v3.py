import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from joypy import joyplot


np.random.seed(42)  # for reproducibility
contracts = ['Contract_A', 'Contract_B', 'Contract_C', 'Contract_D']
data = {contract: pd.Series(np.random.lognormal(mean=0.1, sigma=0.2, size=1000)) 
        for contract in contracts}
df = pd.DataFrame(data)

summary_stats = df.agg(['mean', 'median', 'std', 'skew', 'kurt']).T
summary_stats.columns = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis']
print(summary_stats)

plt.figure(figsize=(16, 12))
sns.set(style="whitegrid")
g = sns.FacetGrid(df.melt(), col="variable", col_wrap=2, height=4, aspect=1.5)
g.map(sns.histplot, "value", kde=True)
g.set_axis_labels("Implied Volatility", "Count")
g.set_titles("{col_name}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for contract in df.columns:
    sorted_data = np.sort(df[contract])
    yvals = np.arange(len(sorted_data))/float(len(sorted_data))
    plt.plot(sorted_data, yvals, label=contract)
plt.title('Cumulative Distribution of Implied Volatility')
plt.xlabel('Implied Volatility')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#VERY NICE!
plt.figure(figsize=(12, 8))
joyplot(
    data=df,
    title="Implied Volatility Distributions",
    labels=df.columns,
    colormap=plt.cm.viridis,
    overlap=0.7
)
plt.xlabel("Implied Volatility")
plt.show()

def distribution_difference(x, y):
    return stats.ks_2samp(x, y).statistic

diff_matrix = df.apply(lambda x: df.apply(lambda y: distribution_difference(x, y)))

plt.figure(figsize=(10, 8))
sns.heatmap(diff_matrix, annot=True, cmap='YlOrRd')
plt.title('Heatmap of Distribution Differences (K-S Statistic)')
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, contract in enumerate(df.columns):
    stats.probplot(df[contract], dist="norm", plot=axes[i])
    axes[i].set_title(f'Q-Q Plot: {contract}')

plt.tight_layout()
plt.show()


