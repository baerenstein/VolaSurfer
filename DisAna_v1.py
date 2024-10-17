
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

iv_series = pd.Series(np.random.lognormal(mean=0.1, sigma=0.2, size=1000))

mean = iv_series.mean()
median = iv_series.median()
std_dev = iv_series.std()
skewness = iv_series.skew()
kurtosis = iv_series.kurtosis()

print(f"Mean: {mean:.4f}")
print(f"Median: {median:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

plt.figure(figsize=(10, 6))

# Histogram
plt.hist(iv_series, bins=50, density=True, alpha=0.7, color='skyblue')

# Kernel Density Estimation
kde = stats.gaussian_kde(iv_series)
x_range = np.linspace(iv_series.min(), iv_series.max(), 100)
plt.plot(x_range, kde(x_range), 'r-', lw=2)

# Normal distribution for comparison
plt.plot(x_range, stats.norm.pdf(x_range, mean, std_dev), 'g--', lw=2)

plt.title('Distribution of Implied Volatility')
plt.xlabel('Implied Volatility')
plt.ylabel('Density')
plt.legend(['KDE', 'Normal Dist.', 'Histogram'])
plt.grid(True, alpha=0.3)
plt.show()