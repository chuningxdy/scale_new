"""
Track the total expected weight norm and loss at each training step for a
given (N, B, K) under the NQS model with adaptive LR (weight norm regularization),
matching compute_nqs_regularized.

Usage:
    python track_weight_norm.py
"""

import csv
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from nqs import compute_nqs_regularized

jax.config.update("jax_enable_x64", True)

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

INIT_LR = 1.999
INIT_WEIGHT_NORM_SQUARED_FN = lambda N: N * 0.03**2


def compute_weight_norm_trajectory(N, B, K, nqs_params, interval=None):
    """
    Compute total weight norm and loss at each interval step, using
    adaptive LR (weight norm regularization) matching the NQS package.

    Parameters
    ----------
    N, B, K : int
        Model size, batch size, number of training steps.
    nqs_params : dict
        NQS parameters {p, q, P, Q, e_irr, R, r}.
    interval : int or None
        Step interval for LR adaptation. If None, uses same logic as NQS package.

    Returns
    -------
    steps : array
        Cumulative step count at each checkpoint.
    weight_norms : array
        sqrt(init_weight_norm_squared + w2) at each checkpoint.
    losses : array
        e_bias + e_var + e_appx + e_irr at each checkpoint.
    """
    p = nqs_params['p']
    q = nqs_params['q']
    P = nqs_params['P']
    Q = nqs_params['Q']
    R = nqs_params['R']
    r = nqs_params['r']
    e_irr = nqs_params['e_irr']

    init_wn_sq = INIT_WEIGHT_NORM_SQUARED_FN(N)

    # Match NQS package interval logic
    if interval is None:
        if K > 10:
            interval = max(K // 100, min(1000, K // 10))
        else:
            interval = 1

    ns = jnp.arange(1, N + 1, dtype=jnp.float64)
    lambda_n = Q / ns**q
    w_star2_n = (P / Q) / ns**(p - q)
    init_R_n = 0.5 * P / ns**p
    noise_R_n = R / ns**r
    e_appx = 0.5 * P * float(jax.scipy.special.zeta(p, N + 1))

    @jax.jit
    def _step(b_factor, v_factor, lr, num_steps):
        """Advance b_factor and v_factor by num_steps at given lr."""
        a_n = 1.0 - lr * lambda_n
        a_n_sq = a_n ** 2
        prod_factor = a_n_sq ** num_steps
        sumprod_factor = (1.0 - prod_factor) / (1.0 - a_n_sq)

        b_factor_new = b_factor * prod_factor
        v_factor_new = v_factor * prod_factor + lr**2 / B * sumprod_factor

        return b_factor_new, v_factor_new

    @jax.jit
    def _evaluate(b_factor, v_factor):
        """Compute w2, e_bias, e_var from current factors."""
        e_bias_n = init_R_n * b_factor
        e_var_n = 0.5 * lambda_n * noise_R_n * v_factor

        w2_n = (1.0 - 2.0 * jnp.sqrt(b_factor)) * w_star2_n + 2.0 / lambda_n * (e_bias_n + e_var_n)
        w2 = jnp.sum(w2_n)
        e_bias = jnp.sum(e_bias_n)
        e_var = jnp.sum(e_var_n)

        return w2, e_bias, e_var

    # Initial state: b_factor=1, v_factor=0 for all dimensions
    b_factor = jnp.ones(N, dtype=jnp.float64)
    v_factor = jnp.zeros(N, dtype=jnp.float64)

    # Record initial state
    w2_init, e_bias_init, e_var_init = _evaluate(b_factor, v_factor)
    w2_val = float(w2_init)

    cum_steps = [0]
    weight_norms = [np.sqrt(init_wn_sq + w2_val)]
    losses = [float(e_bias_init) + float(e_var_init) + e_appx + e_irr]

    init_lr = INIT_LR
    curr_lr = INIT_LR
    k_done = 0

    while k_done < K:
        # Adaptive LR scaling
        if w2_val < 0.0:
            lr_scale = 1.0
        else:
            lr_scale = 1.0 - w2_val / (init_wn_sq + w2_val)

        step_lr = min(curr_lr, init_lr * lr_scale)
        curr_lr = step_lr

        num_steps = min(interval, K - k_done)

        b_factor, v_factor = _step(b_factor, v_factor, step_lr, num_steps)
        w2_val_jax, e_bias_val, e_var_val = _evaluate(b_factor, v_factor)
        w2_val = float(w2_val_jax)

        k_done += num_steps
        cum_steps.append(k_done)
        weight_norms.append(np.sqrt(init_wn_sq + w2_val))
        losses.append(float(e_bias_val) + float(e_var_val) + e_appx + e_irr)

    return np.array(cum_steps), np.array(weight_norms), np.array(losses)


def compute_weight_norm_trajectory_standard(N, B, K, nqs_params, n_checkpoints=100):
    """
    Compute weight norm and loss trajectory with constant LR (no regularization),
    matching compute_nqs_standard.
    """
    p = nqs_params['p']
    q = nqs_params['q']
    P = nqs_params['P']
    Q = nqs_params['Q']
    R = nqs_params['R']
    r = nqs_params['r']
    e_irr = nqs_params['e_irr']
    gamma = INIT_LR

    init_wn_sq = INIT_WEIGHT_NORM_SQUARED_FN(N)

    steps = np.unique(np.linspace(0, K, n_checkpoints, dtype=int))
    ns = jnp.arange(1, N + 1, dtype=jnp.float64)

    lambda_n = Q / ns**q
    w_star2_n = (P / Q) / ns**(p - q)
    init_R_n = 0.5 * P / ns**p
    noise_R_n = R / ns**r
    a_n = 1.0 - gamma * lambda_n
    a_n_sq = a_n ** 2
    e_appx = 0.5 * P * float(jax.scipy.special.zeta(p, N + 1))

    @jax.jit
    def _at_k(k):
        b_factor = a_n ** (2 * k)
        v_factor = (1.0 - b_factor) / (1.0 - a_n_sq) * gamma**2 / B

        e_bias_n = init_R_n * b_factor
        e_var_n = 0.5 * lambda_n * noise_R_n * v_factor

        w2_n = (1.0 - 2.0 * jnp.sqrt(b_factor)) * w_star2_n + 2.0 / lambda_n * (e_bias_n + e_var_n)
        w2 = jnp.sum(w2_n)
        loss = jnp.sum(e_bias_n + e_var_n) + e_appx + e_irr

        return w2, loss

    weight_norms = []
    losses = []
    for k in steps:
        w2, loss = _at_k(k)
        weight_norms.append(np.sqrt(init_wn_sq + float(w2)))
        losses.append(float(loss))

    return np.array(steps), np.array(weight_norms), np.array(losses)


# ------------------------------------------------------------------ #
#  Test case                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    N, B, K = 10_000_000, 96, 1250

    # Load LLM data
    LLM_CSV = "../outputs/test_run_model/loss_curve_df.csv"
    llm_steps, llm_losses, llm_wn = [], [], []
    with open(LLM_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            llm_steps.append(int(row['ckpt']))
            llm_losses.append(float(row['loss']))
            llm_wn.append(float(row['weight_norm']))
    llm_steps = np.array(llm_steps)
    llm_losses = np.array(llm_losses)
    llm_wn = np.array(llm_wn)

    # Compute NQS trajectories
    init_wn = INIT_WEIGHT_NORM_SQUARED_FN(N)
    print(f"Tracking weight norm: N={N}, B={B}, K={K}")

    steps, weight_norms, losses = compute_weight_norm_trajectory(N, B, K, NQS_PARAMS)
    steps_std, weight_norms_std, losses_std = compute_weight_norm_trajectory_standard(N, B, K, NQS_PARAMS)

    # Reconcile with NQS package
    h = {'N': N, 'K': K, 'B': B}
    nqs_pkg_risk = float(compute_nqs_regularized(NQS_PARAMS, h, INIT_WEIGHT_NORM_SQUARED_FN))

    init_wn_norm = np.sqrt(init_wn)
    print(f"  Init weight norm:          {init_wn_norm:.4f}")
    print(f"  NQS weight norm at k=0:    {weight_norms[0]:.4f}")
    print(f"  NQS weight norm at k={K}:  {weight_norms[-1]:.4f}")
    print(f"  LLM weight norm at k={K}:  {llm_wn[-1]:.4f}")
    print(f"  NQS loss (this script):    {losses[-1]:.6f}")
    print(f"  NQS loss (package):        {nqs_pkg_risk:.6f}")
    print(f"  Loss difference:           {abs(losses[-1] - nqs_pkg_risk):.2e}")
    print(f"  Loss rel difference:       {abs(losses[-1] - nqs_pkg_risk) / nqs_pkg_risk:.2e}")
    print(f"  LLM loss at k={K}:         {llm_losses[-1]:.6f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Weight norm
    ax1.plot(steps, weight_norms, linewidth=2, label='NQS (regularized)')
    ax1.plot(steps_std, weight_norms_std, linewidth=2, linestyle='--', label='NQS (standard)')
    ax1.plot(llm_steps, llm_wn, 'o-', color='red', markersize=4, linewidth=1.5, label='LLM')
    ax1.axhline(init_wn_norm, color='gray', linestyle='--', linewidth=1, label=f'Init ({init_wn_norm:.2f})')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('||w||')
    ax1.set_title('Weight Norm', fontweight='bold')
    ax1.legend(fontsize=7)

    # Loss
    ax2.plot(steps, losses, linewidth=2, label='NQS (regularized)')
    ax2.plot(steps_std, losses_std, linewidth=2, linestyle='--', label='NQS (standard)')
    ax2.plot(llm_steps, llm_losses, 'o-', color='red', markersize=4, linewidth=1.5, label='LLM')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss', fontweight='bold')
    ax2.legend(fontsize=7)

    fig.suptitle(f'N={N}, B={B}, K={K}', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('weight_norm_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved weight_norm_trajectory.png")
