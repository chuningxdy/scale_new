"""
Compute the mean and variance of the loss under the NQS model, assuming
the loss is a weighted sum of squared Gaussians across dimensions.

Uses constant learning rate (no adaptive weight decay), matching
compute_nqs_standard.

Usage:
    python compute_loss_variance.py
"""

import jax
import jax.numpy as jnp
import jax.scipy.special
from nqs import compute_nqs_standard

# Full-dataset fitted parameters (from demo_fit_and_predict.py)
NQS_PARAMS = {
    'p': 1.1169923126408567,
    'q': 0.5880068752176018,
    'P': 3.581383803474162,
    'Q': 0.9355347298617339,
    'e_irr': 0.44947765223894076,
    'R': 4.282677723527906,
    'r': 1.4850559677444017,
}

GAMMA = 1.999  # constant learning rate, same as nqs._DEFAULT_LR


def _per_dim(n, B, K, p, q, P, Q, R, r, gamma):
    """Compute per-dimension lambda_n, mu_n^2, sigma_n^2."""
    lambda_n = Q / n**q
    w0_sq = (P / Q) / n**(p - q)

    a_n = 1.0 - gamma * lambda_n
    a_n_2K = a_n ** (2 * K)

    mu_n_sq = a_n_2K * w0_sq
    sigma_n_sq = (gamma**2 / B) * (R / n**r) * (1.0 - a_n_2K) / (1.0 - a_n**2)

    return lambda_n, mu_n_sq, sigma_n_sq


# Vectorize over dimension index n
_per_dim_vec = jax.vmap(_per_dim, in_axes=(0, None, None, None, None, None, None, None, None, None))


def compute_loss_mean_and_variance(N, B, K, p, q, P, Q, e_irr, R, r):
    """
    Compute the mean and variance of the loss under the NQS model.

    E[Q] = e_irr + e_appx + 0.5 * sum_{n=1}^{N} lambda_n * (mu_n^2 + sigma_n^2)

    Parameters are scalars. N must be a concrete Python int.

    Returns
    -------
    mean : float
        E[Q]
    variance : float
        Var[Q]
    """
    gamma = GAMMA
    ns = jnp.arange(1, N + 1, dtype=jnp.float64)

    @jax.jit
    def _compute(ns, B, K, p, q, P, Q, e_irr, R, r):
        e_appx = 0.5 * P * jax.scipy.special.zeta(p, N + 1)
        lambda_n, mu_n_sq, sigma_n_sq = _per_dim_vec(ns, B, K, p, q, P, Q, R, r, gamma)
        mean = e_irr + e_appx + 0.5 * jnp.sum(lambda_n * (mu_n_sq + sigma_n_sq))
        variance = 0.25 * jnp.sum(lambda_n**2 * (2.0 * sigma_n_sq**2 + 4.0 * mu_n_sq * sigma_n_sq))
        return mean, variance

    return _compute(ns, B, K, p, q, P, Q, e_irr, R, r)


# ------------------------------------------------------------------ #
#  Test case: (N, B, K) = (1e6, 463, 2838)                           #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    N, B, K = 1_000_000, 463, 2838
    h = {'N': N, 'K': K, 'B': B}

    # Reference: NQS standard prediction (constant LR, no weight decay)
    nqs_risk = float(compute_nqs_standard(NQS_PARAMS, h))
    print(f"Test case: N={N}, B={B}, K={K}")
    print(f"  NQS risk (standard): {nqs_risk:.6f}")

    # Our computation
    mean, variance = compute_loss_mean_and_variance(
        N, B, K,
        NQS_PARAMS['p'], NQS_PARAMS['q'], NQS_PARAMS['P'], NQS_PARAMS['Q'],
        NQS_PARAMS['e_irr'], NQS_PARAMS['R'], NQS_PARAMS['r'],
    )
    mean, variance = float(mean), float(variance)
    print(f"  Computed mean:       {mean:.6f}")
    print(f"  Difference:          {abs(mean - nqs_risk):.2e}")
    print(f"  Rel difference:      {abs(mean - nqs_risk) / nqs_risk:.2e}")
    print(f"  Mean matches NQS:    {abs(mean - nqs_risk) / nqs_risk < 1e-3}")
    print(f"  Computed variance:   {variance:.6f}  (placeholder)")
