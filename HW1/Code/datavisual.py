import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom

# Set the style for seaborn
sns.set(style="whitegrid")

def plot_binomial_pmf(M):
    # Parameters
    n = M
    p = 0.5
    k = np.arange(0, n+1)
    pmf = binom.pmf(k, n, p)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    sns.barplot(x=k, y=pmf, color='skyblue', edgecolor='black')
    plt.title(f'PMF of X (Number of Right Turns) for M = {M}')
    plt.xlabel('Number of Right Turns (k)')
    plt.ylabel('Probability P(X = k)')
    plt.xticks(k)
    plt.ylim(0, pmf.max() + 0.05)
    plt.show()

# Plot for M = 5
plot_binomial_pmf(5)

# Plot for M = 10
plot_binomial_pmf(10)

# Plot for M = 100
# For M=100, to visualize the PMF effectively, we can use a line plot
def plot_binomial_pmf_large(M):
    n = M
    p = 0.5
    k = np.arange(0, n+1)
    pmf = binom.pmf(k, n, p)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k, pmf, 'o', markersize=4, label='Binomial PMF')
    
    # Overlay the normal distribution approximation
    mu = n * p
    sigma = np.sqrt(n * p * (1 - p))
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2)),
             'r-', label='Normal Approximation')
    
    plt.title(f'PMF of X for M = {M} with Normal Approximation')
    plt.xlabel('Number of Right Turns (k)')
    plt.ylabel('Probability P(X = k)')
    plt.legend()
    plt.show()

plot_binomial_pmf_large(100)
