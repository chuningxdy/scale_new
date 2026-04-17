"""
Compare Gamma vs Gaussian approximations of the loss distribution
using the mean and variance computed by compute_loss_variance.py.

Usage:
    python visualize_loss_distribution.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
from compute_loss_variance import compute_loss_mean_and_variance, NQS_PARAMS

import jax
jax.config.update("jax_enable_x64", True)

N, B, K = 1_000_000, 463, 2838

mean, var = compute_loss_mean_and_variance(
    N, B, K,
    NQS_PARAMS['p'], NQS_PARAMS['q'], NQS_PARAMS['P'], NQS_PARAMS['Q'],
    NQS_PARAMS['e_irr'], NQS_PARAMS['R'], NQS_PARAMS['r'],
)
mean, var = float(mean), float(var)
std = np.sqrt(var)

print(f"Mean: {mean:.6f}, Var: {var:.6f}, Std: {std:.6f}")

# Gamma: shape k = mean^2 / var, scale theta = var / mean
k = mean**2 / var
theta = var / mean
print(f"Gamma shape k={k:.2f}, scale theta={theta:.6f}")

gamma_dist = gamma(a=k, scale=theta)
norm_dist = norm(loc=mean, scale=std)

# Plot range: mean +/- 5 std
x = np.linspace(mean - 5 * std, mean + 5 * std, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# PDF comparison
ax1.plot(x, gamma_dist.pdf(x), label='Gamma', linewidth=2)
ax1.plot(x, norm_dist.pdf(x), label='Gaussian', linewidth=2, linestyle='--')
ax1.axvline(mean, color='gray', linestyle=':', linewidth=0.8, label='Mean')
ax1.set_xlabel('Loss')
ax1.set_ylabel('Density')
ax1.set_title('PDF Comparison', fontweight='bold')
ax1.legend()

# CDF comparison (focus on tails)
ax2.plot(x, gamma_dist.cdf(x), label='Gamma', linewidth=2)
ax2.plot(x, norm_dist.cdf(x), label='Gaussian', linewidth=2, linestyle='--')
ax2.axvline(mean, color='gray', linestyle=':', linewidth=0.8)
ax2.set_xlabel('Loss')
ax2.set_ylabel('CDF')
ax2.set_title('CDF Comparison', fontweight='bold')
ax2.legend()

plt.suptitle(f'N={N}, B={B}, K={K} | mean={mean:.4f}, std={std:.4f}, shape={k:.0f}',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('gamma_vs_gaussian.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved gamma_vs_gaussian.png")

# Print quantile comparison
print(f"\nQuantile comparison:")
print(f"  {'%':>6s} {'Gamma':>12s} {'Gaussian':>12s} {'Diff':>12s}")
for q in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]:
    g = gamma_dist.ppf(q)
    n = norm_dist.ppf(q)
    print(f"  {q*100:>5.0f}% {g:>12.6f} {n:>12.6f} {g-n:>12.6f}")
