import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
if not (jnp.array(1.).dtype is jnp.dtype("float64")):
    raise RuntimeError("jax.config.jax_enable_x64 must be set for numerical stability. "
                      "Please set this environment variable before using NQS.")
from collections import namedtuple
from functools import partial
import os
import time
import datetime
import json
from jax import lax

import numpy as np
from pyDOE import lhs  # for latin hypercube sampling

import pandas as pd
from omegaconf import OmegaConf
import yaml

 

# ----------------------------------------------------#
#       Define the NQS                                #
# ----------------------------------------------------#

#  Q(w) = e + 1/2 \sum_{n=1}^\infty \lambda_n (w_n - w_n^*)^2

#  weight update rules:
#  if n <= N:
#      w_n^(k+1) = w_n^(k) - lr_k \lambda_n (w^(k) - w^*) + lr_k * \sigma_n * z_n^(k)
#  else:
#      w_n^(k+1) = w_n^(k)
#  where \sigma_n = sqrt(R/n^r * 1/B_k), and z_n^(k) are i.i.d. standard guassians

#  \lambda_n = Q/n^q
#  E[ \lambda_n * (w_n^(0)-w^*)^2 ]= P/n^p

#  We can simplify 
#  E[Q(w^(K))] = e_irr + e_appx + e_est
#   where e_appx = 1/2 * \sum_{n=(N+1)}^\infty P/n^p = 1/2 * P * Zeta(x = p, q = N+1)
#        (documentation for the JAX Zeta function: https://docs.jax.dev/en/latest/_autosummary/jax.scipy.special.zeta.html )
# 
#  and
#
#  e_est: we can either simulate, or use the formula below
#   
#        e_est = e_bias + e_var
#        e_bias = 1/2 * sum_{n=1}^N P/n^p * prod_{k=1}^K (1-lr_k*Q/n^q)^2
#        e_var = 1/2 * sum_{n=1}^N Q/n^q * R/n^r * sum_{k=1}^K 1/B_k * lr_k^2 * prod_{j=k+1}^K (1-lr_k*Q/n^q)^2




# ------------------------------------ #
#           Requirements               #
# ------------------------------------ #

#
# ------ Three functions -------
#

# 0. simulate nqs (for checks)
# 1. compute nqs (fast) - "divde and conquer" + 20 fixed point quad + EM
# 2. fit nqs (fast, parallelized) - incl. initialisation, optimization of the 6 params

#
# --------- Features of compute nqs --------
#

# 1. learning rate and BS schedule (step functions) (not available in fitting)
# 2. learning rate adaptation - use optimial learning rate greedy search (not available in fitting)




#  ---------- Preliminary Definitions --------- #

#NQS = namedtuple("NQS", ["p", "q", "P", "Q", "e_irr", "R", "r"])
Cfg = namedtuple("Cfg", ["N", "B", "K", "lr", "sch"])
# sch is a dictionary with 3 fields: decay_at, decay_amt (for lr), B_decay_amt (for batch size)
Step = namedtuple("Step", ["lr", "B", "num_steps"]) # each phase in a step schedule has these 3 attributes


def _get_schedules(lr, B, K, sch):
    '''
    return learning rate and batch size schedules as arrays of length K
    '''
    
    # learning rate schedule
    init_lr = lr
    lrs = jnp.ones((K,)) * init_lr
    for decay_at, decay_amt in zip(sch["decay_at"], sch["decay_amt"]):
        decay_step = int(decay_at * K)
        lrs = lrs.at[decay_step:].set(init_lr * decay_amt)

    # batch size schedule
    init_B = B
    Bs = jnp.ones((K,)) * init_B
    for decay_at, decay_amt in zip(sch["decay_at"], sch["B_decay_amt"]):
        decay_step = int(decay_at * K)
        Bs = Bs.at[decay_step:].set(init_B * decay_amt)

    return lrs, Bs


def _process_schedule_steps(lr, B, K, sch):
    '''
    return a list of (lr, B, num_steps) tuples, representing the schedule segments
    0 <= num_steps <= K
    '''

    init_lr = lr
    init_B = B
    
    steps = []
    prev_step = 0
    for decay_at, decay_amt, B_decay_amt in zip(sch["decay_at"], sch["decay_amt"], sch["B_decay_amt"]):
        decay_step = int(decay_at * K)
        num_steps = decay_step - prev_step
        steps.append(jnp.array([lr, B, num_steps]))
        lr = init_lr * decay_amt
        B = init_B * B_decay_amt
        prev_step = decay_step

    # final segment
    num_steps = K - prev_step
    steps.append(jnp.array([lr, B, num_steps]))

    return jnp.array(steps)




def _e_irr(nqs, cfg):
    p, q, P, Q, e_irr, R, r = nqs
    to_return = e_irr
    return to_return

def _e_irr_no_sch(nqs, cfg_array):
    p, q, P, Q, e_irr, R, r = nqs
    return e_irr

def _e_appx(nqs, cfg):
    N = cfg.N
    p, q, P, Q, e_irr, R, r = nqs
    to_return =  0.5 * P * jnp.squeeze(jax.scipy.special.zeta(p, N+1))
    return to_return
    

def _e_appx_no_sch(nqs, cfg_array):
    N = cfg_array[0]
    p, q, P, Q, e_irr, R, r = nqs
    return 0.5 * P * jnp.squeeze(jax.scipy.special.zeta(p, N+1))

# ---- Simulation of Bias & Variance: for testing ---- #

def _norm_w(w):
    return jnp.sqrt(jnp.sum(w**2))

def _e_est_simulate(nqs, cfg, key, max_w_norm = jnp.inf, save_traj = False):

    N = cfg.N
    K = cfg.K
    B = cfg.B
    lr = cfg.lr
    sch = cfg.sch

    p, q, P, Q, e_irr, R, r = nqs

    # initialize weights to be N dim vector
    key, key_init = jax.random.split(key) 
    w_rand = jax.random.normal(key_init, shape=(N,)) 
    w_rand = 1.0 # set to fixed to reduce variance in testing; using the random version requires more sims to converge
    w_star = w_rand * jnp.sqrt(P / jnp.arange(1, N+1)**p / Q * jnp.arange(1, N+1)**q)
    w0 = jnp.zeros_like(w_star)
    #w0 = w_rand * jnp.sqrt(P / jnp.arange(1, N+1)**p / Q * jnp.arange(1, N+1)**q)
    #w_star = jnp.zeros_like(w0)
    w = w0.copy()
    keys = jax.random.split(key, K)
    lrs, Bs = _get_schedules(lr, B, K, sch)

    if save_traj:
        e_traj = []
        w_norm_traj = []

    for k in range(K):
        lr_k = lrs[k]
        B_k = Bs[k]
        z_k = jax.random.normal(keys[k], shape=(N,))
        lambda_n = Q / jnp.arange(1, N+1)**q
        sigma_n = jnp.sqrt(R / jnp.arange(1, N+1)** r * 1.0 / B_k)
        w = w - lr_k * lambda_n * (w - w_star) + lr_k * sigma_n * z_k 
        w_norm = _norm_w(w)
        if w_norm > max_w_norm:
            w = w / w_norm * max_w_norm
        if save_traj:
            e_k = 0.5 * jnp.sum( Q / jnp.arange(1, N+1)** q * (w - w_star)**2)
            e_traj.append(e_k)
            w_norm_traj.append(w_norm)
    e_est = 0.5 * jnp.sum(Q / jnp.arange(1, N+1)** q * (w - w_star)**2)
    
    
    if save_traj:
        return e_est, jnp.array(e_traj), jnp.array(w_norm_traj)

    return e_est

def _plot_traj(e_traj, w_norm_traj, K):
        # make a plot of e_traj and w_norm_traj
        # the x axis is the step number
        # use log-log scale for e_traj and w_norm_traj
        # side-by-side plots
        import matplotlib.pyplot as plt

        # make moving average for e_traj
        e_traj_ma = jnp.convolve(e_traj, jnp.ones(20)/20, mode='valid')
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(jnp.arange(1, K+1), jnp.array(e_traj))
        plt.plot(jnp.arange(20, K+1), jnp.array(e_traj_ma), color='red', label='Moving Average (window=20)')
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Step")
        plt.ylabel("Estimated e_est")
        plt.title("Trajectory of e_est during simulation")
        plt.subplot(1, 2, 2)
        plt.plot(jnp.arange(1, K+1), jnp.array(w_norm_traj))
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Step")
        plt.ylabel("Norm of w")
        plt.title("Trajectory of ||w|| during simulation")
        plt.tight_layout()
        # save the plot
        plt.savefig("e_est_w_norm_trajectory.png")

        return None






def _e_est_simulate_many(nqs, cfg, key, num_sims=10):
    e_est_sims = []
    for sim in range(num_sims):
        key, subkey = jax.random.split(key)
        e_est_sim = _e_est_simulate(nqs, cfg, subkey)
        e_est_sims.append(e_est_sim)
    return jnp.array(e_est_sims)

def _e_est_simulate_mean(nqs, cfg, key, num_sims=10):
    e_est_sims = _e_est_simulate_many(nqs, cfg, key, num_sims)
    e_est_mean = jnp.mean(e_est_sims)
    return e_est_mean



# -------- Exact Computation of Bias & Variance, for testing -------- #

def _e_est_formula(nqs, cfg, return_bias_var=False):

    N = cfg.N
    K = cfg.K
    B = cfg.B
    lr = cfg.lr
    sch = cfg.sch

    lrs, Bs = _get_schedules(lr, B, K, sch)

    p, q, P, Q, e_irr, R, r = nqs


    n = jnp.arange(1, N+1)
    lambda_n = Q / n**q

    # bias term
    terms = (1.0 - lrs[:, None] * lambda_n[None, :])**2
    e_bias = 0.5 * jnp.sum(P / n**p * jnp.prod(terms, axis=0))

    # variance term
    prods = jnp.cumprod(terms[::-1, :], axis=0)[::-1, :]/terms[0:, :]
    e_var = 0.5 * jnp.sum(lambda_n * (R / n**r) * jnp.sum((lrs**2 / Bs)[:, None] * prods, axis=0))

    e_est = e_bias + e_var
    if return_bias_var:
        return e_est, e_bias, e_var
    return e_est

def _grad_e_est_formula(nqs, cfg):

    def e_est_wrap(nqs_local):
        return _e_est_formula(nqs_local, cfg)

    grad_e_est = jax.grad(e_est_wrap)(nqs)

    return grad_e_est


# ---------------------------------------------------- #
#         Fast Computation of Bias & Variance          #
# ---------------------------------------------------- #

def _jax_quad(f, a, b):
    """
    JAX implementation of numerical integration using fixed-point Gaussian quadrature
      vectorized over multiple integrals

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
    Returns:
        Approximate integral of f from a to b
    """
    # For efficiency, use simple fixed-size rules
    # This avoids any conditional logic that could cause tracer issues
    
    # Fixed 20-point Gauss-Legendre quadrature for general use
    x = jnp.array([
        -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
        -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
        -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
        -0.0765265211334973, 0.0765265211334973, 0.2277858511416451,
        0.3737060887154195, 0.5108670019508271, 0.6360536807265150,
        0.7463319064601508, 0.8391169718222188, 0.9122344282513259,
        0.9639719272779138, 0.9931285991850949
    ])
    
    w = jnp.array([
        0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
        0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
        0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
        0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
        0.1420961093183820, 0.1316886384491766, 0.1181945319615184,
        0.1019301198172404, 0.0832767415767048, 0.0626720483341091,
        0.0406014298003869, 0.0176140071391521
    ])
    
    # Transform the interval [a, b] to [-1, 1]
    mid = (b + a) / 2
    radius = (b - a) / 2
    
    # Map points from [-1, 1] to [a, b]
    mapped_x = mid + radius * x
    
    # Evaluate function at quadrature points
    y_values = jax.vmap(f)(mapped_x) 
    
    # Compute integral (vectorized over n)
    result = radius * jnp.sum(w[:, jnp.newaxis] * y_values, axis=0)

    return result


def _em(g, L, U, verbose=False):
    """
    First-order Euler-Maclaurin approximation for estimating 
    the sum g(L+1) + g(L+2) + ... + g(U) 
    
    """

    
    time_start = time.time()
    def f(x):
        n = jnp.exp(x)
        dn = jnp.exp(x)
        return g(n) * dn
    
    logL = jnp.log(L)
    logU = jnp.log(U)

    integral = _jax_quad(f, logL, logU)

    # Compute risk
    risk = integral
    risk += (g(U) - g(L)) / 2

    time_end = time.time()
    if verbose:
        print(f"Time for EM from {L} to {U}: {time_end - time_start} seconds")

    return risk 






def _reduce(f, input_dim=1):
    """
    Recursive reduction for summation over n=1 to N; input function returns a vector of length 2.
    """

    def reducedf(N):

        def bodyf(val):
            s, n = val
            return (s + f(n), n-1)

        def condf(val):
            s, n = val
            return n > 0

        risk, _ = jax.lax.while_loop(condf, bodyf, (jnp.array([0.0]*input_dim), N))
        return risk
    time_end = time.time()

    return reducedf

def _geom_sum(a, n, output_prod=False):
    '''
    Return S = 1 + a + a^2 + ... + a^(n-1)
    a is larger than 1.0
    '''
    ans_pow = a ** n
    ans_sum = (1 - ans_pow) / (1 - a) 
    
    return (ans_sum, ans_pow) if output_prod else ans_sum


def _geom_sum_jittable(a, n, output_prod=False, max_bits=64):
    """
    geo sum in case of higher numerical stability requirements
    Compute S = 1 + a + a^2 + ... + a^(n-1) using a fixed-iteration loop
    with masking, optimized so that after k==0 the per-iteration compute is tiny.
    - Differentiable in `a`
    - JIT-friendly (static loop length)
    - Works for scalar or array `a`
    - `n` may be Python int or JAX scalar (nonnegative)

    Args:
      a: scalar or array, broadcastable
      n: nonnegative Python int or jnp scalar; treated as uint64
      output_prod: if True, also return a**n
      max_bits: static iteration cap (use 64 for uint64; 32 if n < 2**32)

    Returns:
      S (and optionally a**n)
    """
    a = jnp.asarray(a)
    k = jnp.asarray(n, dtype=jnp.uint64)

    # Accumulators
    ans_sum = jnp.zeros_like(a)
    ans_pow = jnp.ones_like(a)
    cur_sum = jnp.ones_like(a)  # sum of current block
    cur_pow = a                 # a^(block_len)

    # Done flag: once true, we skip the expensive math via lax.cond
    done = (k == 0)

    def body(_, carry):
        k, done, ans_sum, ans_pow, cur_sum, cur_pow = carry

        # Use current bit only if not done
        take = ((k & jnp.uint64(1)) == 1) & (~done)
        ans_sum = jnp.where(take, ans_sum + ans_pow * cur_sum, ans_sum)
        ans_pow = jnp.where(take, ans_pow * cur_pow,  ans_pow)

        # If not done: do the doubling math; else: skip heavy ops
        def step(c):
            k, done, ans_sum, ans_pow, cur_sum, cur_pow = c
            next_sum = cur_sum + cur_pow * cur_sum   # s -> s*(1+p)
            next_pow = cur_pow * cur_pow             # p -> p*p
            k = k >> jnp.uint64(1)
            return (k, done, ans_sum, ans_pow, next_sum, next_pow)

        def no_step(c):
            k, done, ans_sum, ans_pow, cur_sum, cur_pow = c
            # Keep cur_sum/cur_pow unchanged; just shift k to progress the loop.
            k = k >> jnp.uint64(1)
            return (k, done, ans_sum, ans_pow, cur_sum, cur_pow)

        k, done, ans_sum, ans_pow, cur_sum, cur_pow = lax.cond(
            done, no_step, step, (k, done, ans_sum, ans_pow, cur_sum, cur_pow)
        )

        # Update done for next iteration
        done = done | (k == 0)

        return (k, done, ans_sum, ans_pow, cur_sum, cur_pow)

    # Fixed iterations; after done==True, each iter is minimal work
    k, done, ans_sum, ans_pow, _, _ = lax.fori_loop(
        0, max_bits, body, (k, done, ans_sum, ans_pow, cur_sum, cur_pow)
    )

    return (ans_sum, ans_pow) if output_prod else ans_sum

def _geom_sum_not_jittable(a, n, output_prod=False):
    """
    geo sum in case of higher numerical stability requirements
    Return S = 1 + a + a^2 + ... + a^(n-1), in O(log n) time.
    a: scalar (int/float/complex) or array (broadcasted)
    n: nonnegative python int or jnp scalar
    Optional output_prod: if True, also return a^n
    """
    a = jnp.asarray(a)
    k = jnp.asarray(n, dtype=jnp.uint64)

    # Accumulators:
    ans_sum = jnp.zeros_like(a)  # accumulated sum so far
    ans_pow = jnp.ones_like(a)   # accumulated a^m factor for appended blocks

    # Current block for length 1: sum=1, pow=a
    cur_sum = jnp.ones_like(a)   # s = 1 + a + ... + a^(len-1)
    cur_pow = a                  # p = a^(len)

    def cond(state):
        k, *_ = state
        return k > 0

    def body(state):
        k, ans_sum, ans_pow, cur_sum, cur_pow = state
        take = jnp.bitwise_and(k, jnp.uint64(1)) == 1

        # If current bit is 1, append this block to the answer
        ans_sum = jnp.where(take, ans_sum + ans_pow * cur_sum, ans_sum)
        ans_pow = jnp.where(take, ans_pow * cur_pow, ans_pow)

        # Double the block: (s, p) -> (s*(1+p), p*p)
        cur_sum = cur_sum + cur_pow * cur_sum
        cur_pow = cur_pow * cur_pow

        # Next bit
        k = jnp.right_shift(k, jnp.uint64(1))
        return (k, ans_sum, ans_pow, cur_sum, cur_pow)

    _, ans_sum, ans_pow, _, _ = lax.while_loop(cond, body, (k, ans_sum, ans_pow, cur_sum, cur_pow))
    if output_prod:
        return ans_sum, ans_pow
    return ans_sum



def _e_dim_bv_steps(nqs, n, steps):
    '''
    Accumulate the factors for both e_bias and e_var at dimension n, over the phases in a multi-step schedule
    '''

    b_factor = 1.0
    v_factor = 0.0

    p, q, P, Q, e_irr, R, r = nqs

    for step in steps:
        lr = step[0]
        B = step[1]
        num_steps = step[2]
        a = 1.0 - lr * (Q / n**q)

        sumprod_factor, prod_factor = _geom_sum(a**2, num_steps, output_prod = True)
        
        b_factor = b_factor * prod_factor
        v_factor = v_factor * prod_factor + lr**2/B * sumprod_factor

    return b_factor, v_factor



def _f(nqs, n, steps):
    p, q, P, Q, e_irr, R, r = nqs
    init_R_n = 0.5 * P / n**p
    lambda_n = Q / n**q
    b_factor, v_factor = _e_dim_bv_steps(nqs, n, steps)
    e_bias_n = init_R_n * b_factor
    e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
    return jnp.array([e_bias_n, e_var_n])

start_time = time.time()
_jit_reduce_f = jax.jit(lambda nqs, N, steps: _reduce(lambda n: _f(nqs, n, steps), input_dim=2)(N))
end_time = time.time()
print(f"Time for JIT compiling reduce f: {end_time - start_time} seconds")

start_time = time.time()
_jit_em_f = jax.jit(lambda L, U, nqs, steps: _em(lambda n: _f(nqs, n, steps), L, U))
end_time = time.time()
print(f"Time for JIT compiling EM f: {end_time - start_time} seconds")

def _e_bias_var_fast(nqs, cfg):
    '''
    Fast computation of e_bias and e_var using EM approximation
    (combining the computation of bias and var to save time)
    '''
    N = cfg.N
    K = cfg.K
    B = cfg.B
    lr = cfg.lr

    sch = cfg.sch
    steps = _process_schedule_steps(lr, B, K, sch)

    #p, q, P, Q, e_irr, R, r = nqs

    #def g(n):
    #    init_R_n = 0.5 * P / n**p
    #    lambda_n = Q / n**q
    #    b_factor, v_factor = _e_dim_bv_steps(nqs, n, steps)
    #    e_bias_n = init_R_n * b_factor
    #    e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
    #    return jnp.array([e_bias_n, e_var_n])

    # use exact summation for first M terms, and EM for the rest
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)
    #start_time = time.time()
    #_jit_reduce_g = jax.jit(
        #_reduce(g, input_dim=2))
    #end_time = time.time()
   # print(f"Time for JIT compiling reduce g: {end_time - start_time} seconds")

    start_time = time.time()
    e_bv_1_to_M = _jit_reduce_f(nqs, M, steps)
    end_time = time.time()
    print(f"Time for computing e_bv_1_to_M: {end_time - start_time} seconds")
    
   # start_time = time.time()
   # _jit_em_g = jax.jit(lambda L, U: _em(g, L, U))
    #end_time = time.time()
   # print(f"Time for JIT compiling EM g: {end_time - start_time} seconds")
    start_time = time.time()
    e_bv_Mplus1_to_N = _jit_em_f(M, N, nqs, steps)
    end_time = time.time()
    print(f"Time for computing e_bv_Mplus1_to_N: {end_time - start_time} seconds")
    e_bv = e_bv_1_to_M + e_bv_Mplus1_to_N


    return e_bv

def _e_bias_var_fast_no_sch(nqs, cfg_array):
    '''
    Fast computation of e_bias and e_var using EM approximation
    (combining the computation of bias and var to save time)
    '''
    N = cfg_array[0]
    K = cfg_array[1]
    B = cfg_array[2]
    lr = cfg_array[3]
    steps = _process_schedule_steps(lr, B, K, {"decay_at":[], "decay_amt":[], "B_decay_amt":[]})

    #p, q, P, Q, e_irr, R, r = nqs

    #def g(n):
    #    init_R_n = 0.5 * P / n**p
    #    lambda_n = Q / n**q
    #    b_factor, v_factor = _e_dim_bv_steps(nqs, n, steps)
    #    e_bias_n = init_R_n * b_factor
    #    e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
    #    return jnp.array([e_bias_n, e_var_n])

    # use exact summation for first M terms, and EM for the rest
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)
    #start_time = time.time()
    #_jit_reduce_g = jax.jit(
        #_reduce(g, input_dim=2))
    #end_time = time.time()
   # print(f"Time for JIT compiling reduce g: {end_time - start_time} seconds")

    start_time = time.time()
    e_bv_1_to_M = _jit_reduce_f(nqs, M, steps)
    end_time = time.time()
    print(f"Time for computing e_bv_1_to_M: {end_time - start_time} seconds")
    
   # start_time = time.time()
   # _jit_em_g = jax.jit(lambda L, U: _em(g, L, U))
    #end_time = time.time()
   # print(f"Time for JIT compiling EM g: {end_time - start_time} seconds")
    start_time = time.time()
    e_bv_Mplus1_to_N = _jit_em_f(M, N, nqs, steps)
    end_time = time.time()
    print(f"Time for computing e_bv_Mplus1_to_N: {end_time - start_time} seconds")
    e_bv = e_bv_1_to_M + e_bv_Mplus1_to_N

    return e_bv

def _risk(nqs, cfg):
    e_est_bv = _e_bias_var_fast(nqs, cfg)
    e_appx = _e_appx(nqs, cfg)
    e_irr = _e_irr(nqs, cfg)
    risk = e_est_bv[0] + e_est_bv[1] + e_appx + e_irr
    return risk

def _risk_no_sch(nqs, cfg_array):
    e_est_bv = _e_bias_var_fast_no_sch(nqs, cfg_array)
    e_appx = _e_appx_no_sch(nqs, cfg_array)
    e_irr = _e_irr_no_sch(nqs, cfg_array)
    risk = e_est_bv[0] + e_est_bv[1] + e_appx + e_irr
    return risk

# ---------------------------------------------------- #
#        Fast Computation of Gradients                 #
# ---------------------------------------------------- #
nqs_dim    = 7  # number of NQS parameters

@jax.jit
def _grad_e_irr(nqs, cfg):
    return jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

@jax.jit
def _grad_e_irr_no_sch(nqs, cfg_array):
    return jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

@jax.jit
def _grad_e_appx(nqs, cfg):
    outputs = jax.grad(lambda nqs_local: _e_appx(nqs_local, cfg))(nqs)
    return outputs  

@jax.jit
def _grad_e_appx_no_sch(nqs, cfg_array):
    outputs = jax.grad(lambda nqs_local: _e_appx_no_sch(nqs_local, cfg_array))(nqs)
    return outputs

def _g(nqs, n, steps):
        p, q, P, Q, e_irr, R, r = nqs
        init_R_n = 0.5 * P / n**p
        lambda_n = Q / n**q
        b_factor, v_factor = _e_dim_bv_steps(nqs, n, steps)
        e_bias_n = init_R_n * b_factor
        e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
        # output a scalar
        return jnp.squeeze(e_bias_n + e_var_n)

@jax.jit
def _grad_g(nqs, n, steps):
        start_time = time.time()
        grad_nqs = jax.grad(lambda nqs_local: _g(nqs_local, n, steps))(nqs)
        end_time = time.time()
        print(f"Grad....Time for computing grad_g: {(end_time - start_time, 4)} seconds")
        return grad_nqs

#@jax.jit
#def _reduced_grad_g(nqs, N, steps):
 ##   start_time = time.time()
  #  out_func = _reduce(lambda n: _grad_g(nqs, n, steps), input_dim=nqs_dim)(N)
  #  end_time = time.time()
  #  print(f"Grad....Time for JIT reduced_grad_g: {(end_time - start_time, 4)} seconds")
  #  return out_func

start_time = time.time()
_reduced_grad_g = jax.jit(lambda nqs, N, steps: _reduce(lambda n: _grad_g(nqs, n, steps), input_dim=nqs_dim)(N))
end_time = time.time()
print(f"Grad.....Time for JIT compiling reduced_grad_g: {(end_time - start_time, 4)} seconds")

start_time = time.time()
_em_grad_g = jax.jit(lambda L, U, nqs, steps: _em(lambda n: _grad_g(nqs, n, steps), L, U))
end_time = time.time()
print(f"Grad.....Time for JIT compiling EM grad_g: {(end_time - start_time, 4)} seconds")


def _grad_e_bias_var_fast(nqs, cfg):
    '''
    Fast computation of gradients of e_bias and e_var w.r.t. the 6 NQS params using EM approximation
    (combining the computation of bias and var to save time)
    '''

    N = cfg.N
    K = cfg.K
    B = cfg.B
    lr = cfg.lr
    sch = cfg.sch
    steps = _process_schedule_steps(lr, B, K, sch)


    # use exact summation for first M terms, and EM for the rest
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)

    start_time = time.time()
    e_grad_bv_1_to_M = _reduced_grad_g(nqs, M, steps)
    end_time = time.time()
    print(f"Grad....Time for reducing e_grad_bv_1_to_M: {(end_time - start_time, 4)} seconds")
    start_time = time.time()
    e_grad_bv_Mplus1_to_N = _em_grad_g(M, N, nqs, steps)
    end_time = time.time()
    print(f"Grad....Time for EM e_grad_bv_Mplus1_to_N: {(end_time - start_time, 4)} seconds")
    e_bv = e_grad_bv_1_to_M + e_grad_bv_Mplus1_to_N


    return e_bv


def _grad_e_bias_var_fast_no_sch(nqs, cfg_array):
    '''
    Fast computation of gradients of e_bias and e_var w.r.t. the 6 NQS params using EM approximation
    (combining the computation of bias and var to save time)
    '''

    N = cfg_array[0]
    K = cfg_array[1]
    B = cfg_array[2]
    lr = cfg_array[3]
    steps = _process_schedule_steps(lr, B, K, {"decay_at": [], "decay_amt": [], "B_decay_amt": []})


    # use exact summation for first M terms, and EM for the rest
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)

    start_time = time.time()
    e_grad_bv_1_to_M = _reduced_grad_g(nqs, M, steps)
    end_time = time.time()
    print(f"Grad....Time for reducing e_grad_bv_1_to_M: {(end_time - start_time, 4)} seconds")
    start_time = time.time()
    e_grad_bv_Mplus1_to_N = _em_grad_g(M, N, nqs, steps)
    end_time = time.time()
    print(f"Grad....Time for EM e_grad_bv_Mplus1_to_N: {(end_time - start_time, 4)} seconds")
    e_bv = e_grad_bv_1_to_M + e_grad_bv_Mplus1_to_N


    return e_bv


def _grad_risk(nqs, cfg):
    grad_e_est = _grad_e_bias_var_fast(nqs, cfg)
    grad_e_appx = _grad_e_appx(nqs, cfg)
    grad_e_irr = _grad_e_irr(nqs, cfg)
    grad_risk = grad_e_est + grad_e_appx + grad_e_irr
    return grad_risk

def _grad_risk_no_sch(nqs, cfg_array):
    grad_e_est = _grad_e_bias_var_fast_no_sch(nqs, cfg_array)
    grad_e_appx = _grad_e_appx_no_sch(nqs, cfg_array)
    grad_e_irr = _grad_e_irr_no_sch(nqs, cfg_array)
    grad_risk = grad_e_est + grad_e_appx + grad_e_irr
    return grad_risk

# ---- Finite Difference Gradient for testing ---- #

def _finite_diff_grad(nqs, cfg, eps_multiplier = 1e-6):
    '''
    Finite difference gradient check for the risk function
    '''

    grad_fd = jnp.zeros_like(nqs)
    fd_grads = []

    for i in range(len(nqs)):
        eps = eps_multiplier * jnp.abs(nqs[i])
        nqs_plus = nqs.at[i].add(eps)
        risk_plus = _risk(nqs_plus, cfg)
        nqs_minus = nqs.at[i].add(-eps)
        risk_minus = _risk(nqs_minus, cfg)
        grad_fd = grad_fd.at[i].set((risk_plus - risk_minus) / (2 * eps))
        fd_grads.append(grad_fd[i])

    return jnp.array(fd_grads)

# ---------------------------------------------------- #
#        Fitting an NQS                                #
# ---------------------------------------------------- #

def latin_hypercube_initializations(seed, num_inits, param_names, param_ranges, r_equals_q = True):
        """Create initialization points using Latin Hypercube Sampling."""
        # Parameter names in order

        # convert the param_ranges onto the normalized space, using to_x
        # to_x is a function that takes in an NQS object and converts it to dim 6 array
        lower_bounds = [param_range[0] for param_range in param_ranges.values()]
        upper_bounds = [param_range[1] for param_range in param_ranges.values()]
        
        # Generate Latin Hypercube samples in [0, 1] range
        # set seed
        np.random.seed(seed)
        samples = lhs(len(param_names), samples=num_inits)
        
        # Scale samples to the parameter ranges
        init_nqs_list = []
        for i in range(num_inits):
            sample = samples[i]
            param_values = jnp.array([0.0] * len(param_names))
            for j in range(len(param_names)):
                param_values = param_values.at[j].set(lower_bounds[j] + sample[j] * (upper_bounds[j] - lower_bounds[j]))
            init_nqs_list.append(param_values)
        
        # convert to jnp array
        init_nqs_array = jnp.array(init_nqs_list)

        r_index = param_names.index('r')
        q_index = param_names.index('q')
        if r_equals_q:
            init_nqs_array = init_nqs_array.at[:, r_index].set(init_nqs_array[:, q_index])

        return init_nqs_array


def _nqs_to_x(nqs, norm_const = 1e5):
    '''
    Convert nqs parameters to normalized x parameters for optimization
    (the exponents p,q,r are kept the same, while the multipliers P,Q,e_irr,R are log-transformed)
    '''
    log_norm_const = jnp.log(norm_const)
    x = jnp.zeros_like(nqs)
    x = x.at[0].set(nqs[0])  # p
    x = x.at[1].set(nqs[1])  # q
    x = x.at[2].set(jnp.log(nqs[2]) / log_norm_const)  # P
    x = x.at[3].set(jnp.log(nqs[3]) / log_norm_const)  # Q
    x = x.at[4].set(jnp.log(nqs[4]) / log_norm_const)  # e_irr
    x = x.at[5].set(jnp.log(nqs[5]) / log_norm_const)  # R
    x = x.at[6].set(nqs[6])  # r


    return x


def _x_to_nqs(x, norm_const = 1e5):
    '''
    Convert normalized x parameters back to nqs parameters
    '''
    nqs = jnp.zeros_like(x)
    nqs = nqs.at[0].set(x[0])  # p
    nqs = nqs.at[1].set(x[1])  # q
    nqs = nqs.at[2].set(norm_const ** x[2])  # P
    nqs = nqs.at[3].set(norm_const ** x[3])  # Q
    nqs = nqs.at[4].set(norm_const ** x[4])  # e_irr
    nqs = nqs.at[5].set(norm_const ** x[5])  # R
    nqs = nqs.at[6].set(x[6])  # r

    return nqs



def _dnqs_dx(x, nqs, norm_const = 1e5):
    '''
    Compute the Jacobian of nqs with respect to x
    '''
    log_norm_const = jnp.log(norm_const)
    dnqs_dx = jnp.zeros_like(x)
    dnqs_dx = dnqs_dx.at[0].set(1.0)
    dnqs_dx = dnqs_dx.at[1].set(1.0)
    dnqs_dx = dnqs_dx.at[2].set(log_norm_const * nqs[2])
    dnqs_dx = dnqs_dx.at[3].set(log_norm_const * nqs[3])
    dnqs_dx = dnqs_dx.at[4].set(log_norm_const * nqs[4])
    dnqs_dx = dnqs_dx.at[5].set(log_norm_const * nqs[5])
    dnqs_dx = dnqs_dx.at[6].set(1.0)

    return dnqs_dx

def _loss(y_hat, y):
    """
    Log loss function.
    """
    log_y = jnp.log(y)
    log_y_hat = jnp.log(y_hat)
    error = 0.5 * (log_y - log_y_hat) ** 2
    return error

dloss = jax.jit(jax.grad(_loss, argnums=0))


# compute the gradient of the loss with respect to nqs parameters
def _grad_loss_no_sch(nqs, y, cfg_array, tie_r_and_q=True):
    '''
    Compute the gradient of the loss with respect to nqs parameters
    '''
    y_hat = _risk_no_sch(nqs, cfg_array)
    drisk = _grad_risk_no_sch(nqs, cfg_array)
    if tie_r_and_q:
        drisk = drisk.at[6].add(drisk[1])  # add gradient w.r.t q to r
        drisk = drisk.at[1].set(drisk[6])  # set gradient w.r.t q to be the same as r
    grad_loss_nqs = dloss(y_hat, y) * drisk
    return grad_loss_nqs

def _grad_loss_normalized_no_sch(x, y, cfg_array, tie_r_and_q=True):
    '''
    Compute the gradient of the loss with respect to the normalized nqs parameters
    '''
    nqs = _x_to_nqs(x)
    grad_loss_nqs = _grad_loss_no_sch(nqs, y, cfg_array, tie_r_and_q=tie_r_and_q)
    dnqs_dx = _dnqs_dx(x,nqs)
    grad_loss_x = grad_loss_nqs * dnqs_dx

    return grad_loss_x


# use vmap to vectorize _grad_loss_normalized over multiple (y, cfg_array) pairs
# and sum the results across the list
_grad_total_loss_normalized_no_sch_no_tie_weight = lambda x, ys, cfg_arrays: jnp.sum(jax.vmap(lambda x_local, y_local, cfg_local: _grad_loss_normalized_no_sch(x_local, y_local, cfg_local, tie_r_and_q=False), in_axes=(None, 0, 0))(x, ys, cfg_arrays), axis=0)
_grad_total_loss_normalized_no_sch_tie_weight = lambda x, ys, cfg_arrays: jnp.sum(jax.vmap(lambda x_local, y_local, cfg_local: _grad_loss_normalized_no_sch(x_local, y_local, cfg_local, tie_r_and_q=True), in_axes=(None, 0, 0))(x, ys, cfg_arrays), axis=0)

def _fit_nqs(list_of_nqs_inits, cfg_arrays, ys, itrs = 3, gtol=1e-8, return_trajectories=False,
             tie_r_and_q=True):
    
    @jax.jit
    def _calc_total_loss_no_sch(x):
        nqs = _x_to_nqs(x)
        total_loss = 0.0
        for i in range(len(ys)):
            y = ys[i]
            cfg = cfg_arrays[i]
            loss_i = _loss(_risk_no_sch(nqs, cfg), y)
            total_loss = total_loss + loss_i
        return total_loss

    if tie_r_and_q:
        _grad_total_loss_normalized_no_sch = _grad_total_loss_normalized_no_sch_tie_weight
    else:
        _grad_total_loss_normalized_no_sch = _grad_total_loss_normalized_no_sch_no_tie_weight
   
    @jax.jit
    def _calc_grad(x):
        return _grad_total_loss_normalized_no_sch(x, ys, cfg_arrays)

    def _step_fn(carry, step_idx):
        x, m, v, active_flag = carry
        
        # Adam hyperparameters
        learning_rate = 0.001  # Base learning rate
        beta1 = 0.9  # Exponential decay rate for first moment
        beta2 = 0.999  # Exponential decay rate for second moment
        epsilon = 1e-8  # Small constant for numerical stability
        
        grad = _calc_grad(x)  # Use our custom gradient function

        # Apply gradient clipping
        clip_at = 1.0
        clipped_grad = jnp.clip(grad, -clip_at, clip_at)
        
        # Adam update
        # Update biased first moment estimate (momentum term)
        m_new = jnp.where(active_flag, beta1 * m + (1 - beta1) * clipped_grad, m)
        
        # Update biased second moment estimate (velocity term)
        v_new = jnp.where(active_flag, beta2 * v + (1 - beta2) * (clipped_grad ** 2), v)
        
        # Correct bias in first moment
        m_hat = m_new / (1 - beta1 ** (step_idx + 1))
        
        # Correct bias in second moment
        v_hat = v_new / (1 - beta2 ** (step_idx + 1))
        
        # Compute update
        update = learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        # Apply update
        x_new = jnp.where(active_flag, x - update, x)
        
        
        # Check convergence conditions
        grad_norm = jnp.linalg.norm(grad)
        
        # Create convergence flags
        converged_grad = grad_norm < gtol
        
        # Update active flag - we continue if still active and not converged
        new_active_flag = active_flag & ~(converged_grad)
        # Collect data for trajectory
        traj_info = (x_new, grad, grad_norm, new_active_flag)
        
        # Return updated state
        return (x_new, m_new, v_new, new_active_flag), traj_info
    


    @jax.jit
    # Vectorize the step function over multiple initializations
    def _batch_step_fn(batch_state, step_idx):
        batch_results = jax.vmap(lambda state: _step_fn(state, step_idx))(batch_state)
        return batch_results


    

    # Initialize optimization state for each starting point
    list_of_x_inits = [_nqs_to_x(nqs_init) for nqs_init in list_of_nqs_inits]
    init_x = jnp.stack([x for x in list_of_x_inits])
    init_m = jnp.zeros_like(init_x)  # First moment estimate (momentum)
    init_v = jnp.zeros_like(init_x)  # Second moment estimate (velocity)
    init_active_flags = jnp.ones(len(list_of_x_inits), dtype=bool)  # All trajectories start active

    init_state = (init_x, init_m, init_v, init_active_flags)

    # Use scan instead of fori_loop for better differentiability
    @jax.jit
    def scan_step(state_and_traj, i):
        state, traj_list = state_and_traj
        new_state, new_traj_info  = _batch_step_fn(state, i) #
        return (new_state, traj_list), new_traj_info
    
    # Run the optimization with scan and collect trajectories
    (final_state, _), traj_infos = jax.lax.scan( #, traj_infos
        scan_step,
        (init_state, []),  # Initial state
        jnp.arange(itrs)   # Iteration indices
    )

        # Get final results
    final_nqs_params, _, _, _ = final_state
    final_losses = jax.vmap(lambda x: _calc_total_loss_no_sch(x))(final_nqs_params)
    #best_idx = jnp.argmin(final_losses)
    # get best idx but ignore nans
    best_idx = jnp.nanargmin(final_losses)
    best_x = final_nqs_params[best_idx]
    best_nqs = _x_to_nqs(best_x)
    # get the final loss for the best NQS
    best_loss = final_losses[best_idx]
    
    # Convert trajectory info to list
    trajectories = list(traj_infos)

    if return_trajectories:
        return best_nqs, best_loss, best_idx, trajectories
    else:
        return best_nqs, best_loss




# ---------------------------------------------------- #
#        Learning Rate Adaptation                      #
# ---------------------------------------------------- #

# greedy algorithm:
# decays learning rate every n steps by 0.5, 0.25 etc. until loss stops improving
# re-use previous steps for speed

def _merge_schedules(sch, change_points):
    '''
    Merge two learning rate schedules
    '''

    # remove from change_points any points that are already in sch (by decay_at )
    sch_decay_ats = set(sch['decay_at'])
    additional_change_points = [item for item in change_points if item not in sch_decay_ats]
    # label each sch
    sch_labeled = [{**{key: sch[key][j] for key in sch}, 'sch_id': 1} for j in range(len(sch['decay_at']))]
    additional_change_points_labeled = [{'decay_at': item, 'sch_id': 2} for item in additional_change_points]
    #print("sch_labeled:", sch_labeled)
    #print("additional_change_points_labeled:", additional_change_points_labeled)
    # merge by sorting by decay_at
    merged_sch = sorted(sch_labeled + additional_change_points_labeled, key=lambda x: x['decay_at'])
    #p#rint("merged_sch before filling:", merged_sch)
    # rules to interpolate missing fields
    # if this is the first item, fill missing fields of decay_amt and B_decay_amt with 1.0
    # otherwise, fill missing fields with the previous item's value
    for i in range(len(merged_sch)):
        if 'decay_amt' not in merged_sch[i]:
            if i == 0:
                merged_sch[i]['decay_amt'] = 1.0
            else:
                merged_sch[i]['decay_amt'] = merged_sch[i-1]['decay_amt']
        if 'B_decay_amt' not in merged_sch[i]:
            if i == 0:
                merged_sch[i]['B_decay_amt'] = 1.0
            else:
                merged_sch[i]['B_decay_amt'] = merged_sch[i-1]['B_decay_amt']
    #print("merged_sch after filling:", merged_sch)
    # convert the formatted merged_sch back to dict of lists
    merged_sch_dict = {'decay_at': [], 'decay_amt': [], 'B_decay_amt': [], 'sch_id': []}
    for item in merged_sch:
        merged_sch_dict['decay_at'].append(item['decay_at'])
        merged_sch_dict['decay_amt'].append(item['decay_amt'])
        merged_sch_dict['B_decay_amt'].append(item['B_decay_amt'])
        merged_sch_dict['sch_id'].append(item['sch_id'])
    #print("merged_sch_dict:", merged_sch_dict)
    return merged_sch_dict

def _process_schedule_steps_LRA(lr, B, K, sch, interval = 1000):
    '''
    return a list of (lr, B, num_steps, is_change_point) tuples, representing the schedule segments
    0 <= num_steps <= K, where change points are added every 'interval' steps
    is_change_point is 1 if the segment corresponds to a change point, 0 otherwise
    '''

    # get a list of multiples of inteval
    num_changes = K // interval
    change_points = [(i+1) * interval / K for i in range(num_changes)]
    #change_points_sch = [{'decay_at': cp} for cp in change_points]
    sch_with_chg_pts = _merge_schedules(sch, change_points)


    init_lr = lr
    init_B = B
    
    steps = []
    prev_step = 0
    prev_decay_at = 0.0
    for decay_at, decay_amt, B_decay_amt, sch_id in zip(sch_with_chg_pts['decay_at'], sch_with_chg_pts['decay_amt'], sch_with_chg_pts['B_decay_amt'], sch_with_chg_pts['sch_id']):
        decay_step = int(decay_at * K)
        num_steps = decay_step - prev_step
        if prev_decay_at in change_points:
            step_array = jnp.array([lr, B, num_steps, 1])
        else:
            step_array = jnp.array([lr, B, num_steps, 0])
        steps.append(step_array)
        lr = init_lr * decay_amt
        B = init_B * B_decay_amt
        prev_step = decay_step
        prev_decay_at = decay_at

    # final segment
    num_steps = K - prev_step
    if prev_decay_at in change_points:
        step_array = jnp.array([lr, B, num_steps, 1])
    else:
        step_array = jnp.array([lr, B, num_steps, 0])
    steps.append(step_array)
    #print("LRA steps:", steps)
    return jnp.array(steps) # steps are labeled with 1 if they correspond to change points, 0 otherwise



def _e_dim_bv_steps_LRA(nqs, n, steps):
    '''
    Accumulate the factors for both e_bias and e_var at dimension n, over the phases in a multi-step schedule
    '''

    b_factor = 1.0
    v_factor = 0.0

    p, q, P, Q, e_irr, R, r = nqs

    for step in steps:
        lr = step[0]
        B = step[1]
        num_steps = step[2]
        a = 1.0 - lr * (Q / n**q)
        
        sumprod_factor, prod_factor = _geom_sum(a**2, num_steps, output_prod = True)
        
        b_factor = b_factor * prod_factor
        v_factor = v_factor * prod_factor + lr**2/B * sumprod_factor

    return b_factor, v_factor


def _f_LRA(nqs, n, steps):
    p, q, P, Q, e_irr, R, r = nqs
    init_R_n = 0.5 * P / n**p
    lambda_n = Q / n**q
    b_factor, v_factor = _e_dim_bv_steps_LRA(nqs, n, steps)
    e_bias_n = init_R_n * b_factor
    e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
    return jnp.array([e_bias_n, e_var_n])

start_time = time.time()
_jit_reduce_f_LRA = jax.jit(lambda nqs, N, steps: _reduce(lambda n: _f_LRA(nqs, n, steps), input_dim=2)(N))
end_time = time.time()
print(f"Time for JIT compiling reduce f: {end_time - start_time} seconds")

start_time = time.time()
_jit_em_f_LRA = jax.jit(lambda L, U, nqs, steps: _em(lambda n: _f_LRA(nqs, n, steps), L, U))
end_time = time.time()
print(f"Time for JIT compiling EM f: {end_time - start_time} seconds")


def _cosine_lr_with_warmup(k, num_training_steps, warmup_frac=0.01):
    W = int(num_training_steps * warmup_frac)
    T = num_training_steps

    k = jnp.asarray(k)

    warmup_factor = jnp.minimum(1.0, k / max(1, W))
    progress = jnp.maximum(0.0, (k - W) / max(1, T - W))
    cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    

    return  warmup_factor * cosine_factor


def _e_bias_var_LRA(nqs, cfg, interval = 1000, LRA_tol = 0.05, verbose = False):
    '''
    Fast computation of e_bias and e_var using EM approximation
    (combining the computation of bias and var to save time)
    '''
    N = cfg.N
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)


    K_all = cfg.K
    B = cfg.B
    lr = cfg.lr

    sch = cfg.sch
    steps_all = _process_schedule_steps_LRA(lr, B, K_all, sch, interval = interval)


    def _e_bv_step(local_steps):
        e_bv_1_to_M = _jit_reduce_f_LRA(nqs, M, local_steps)
        e_bv_Mplus1_to_N = _jit_em_f_LRA(M, N, nqs, local_steps)
        e_bv = e_bv_1_to_M + e_bv_Mplus1_to_N
        return e_bv

    K = 0
    curr_lr = cfg.lr

    steps_all_pre_LRA = steps_all

    start_time = time.time()
    # iterate through the rest of the steps
    for i in range(0, steps_all.shape[0]):
        
        step = steps_all[i, :]
        step_cgpt_ind = step[3]
        step_lr = step[0]
        # update step learning rate with the minimum of current lr and step lr
        step_lr = jnp.minimum(curr_lr, step_lr)
        curr_lr = step_lr
        steps_all = steps_all.at[i, 0].set(step_lr)

        e_bv_at_step = _e_bv_step(steps_all[0:i+1, :])
        e_bplusv_at_step = e_bv_at_step[0] + e_bv_at_step[1]

        # check if last step is a change point
        if step_cgpt_ind == 0.0: # not a change point
            K = K + step[2]
            if verbose:
                print (f"Step {i}, not a change point,  lr: {step_lr}, e_bv: {e_bv_at_step}, K: {K}")
        else: # change point
            
            # try to decay learning rate by 0.5
            proposed_lr = curr_lr * 0.5
            steps_all = steps_all.at[i, 0].set(proposed_lr)
            e_bv_at_step_proposed = _e_bv_step(steps_all[0:i+1, :])
            e_bplusv_at_step_proposed = e_bv_at_step_proposed[0] + e_bv_at_step_proposed[1]
            # if proposed loss is lower than the current loss, by a tolerance, accept the change
            decay_count = 0
            loss_reduction = e_bplusv_at_step - e_bplusv_at_step_proposed
            loss_reduction = jnp.round(loss_reduction, decimals=5)
            while loss_reduction > LRA_tol and decay_count < 10:
                
                curr_lr = proposed_lr
                e_bv_at_step = e_bv_at_step_proposed
                e_bplusv_at_step = e_bplusv_at_step_proposed
                
                decay_count += 1

                if verbose:
                    print (f"Step {i}, change point, accepted decay, loss reduction: {loss_reduction},  lr: {curr_lr}, e_bv: {e_bv_at_step}, K: {K}")
                            # try to decay learning rate by 0.5
                
                proposed_lr = curr_lr * 0.5
                steps_all = steps_all.at[i, 0].set(proposed_lr)
                e_bv_at_step_proposed = _e_bv_step(steps_all[0:i+1, :])
                e_bplusv_at_step_proposed = e_bv_at_step_proposed[0] + e_bv_at_step_proposed[1]
                loss_reduction = e_bplusv_at_step - e_bplusv_at_step_proposed

            steps_all = steps_all.at[i, 0].set(curr_lr)
            K = K + step[2]
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    if verbose:
        # make a plot of the learning rate schedule and the loss (e_bias + e_var) at each step
        import matplotlib.pyplot as plt
        lrs = steps_all[:, 0]
        lrs_no_LRA = steps_all_pre_LRA[:, 0]

        Ks = []
        change_point_indices = jnp.where(steps_all[:, 3] == 1)[0]
        losses = []
        losses_no_LRA = []
        for i in range(steps_all.shape[0]):

            Ks.append(jnp.sum(steps_all[0:i+1, 2]))

            e_bv_at_step = _e_bv_step(steps_all[0:i+1, :])
            e_bplusv_at_step = e_bv_at_step[0] + e_bv_at_step[1]
            losses.append(e_bplusv_at_step)

            e_bv_at_step_no_LRA = _e_bv_step(steps_all_pre_LRA[0:i+1, :])
            e_bplusv_at_step_no_LRA = e_bv_at_step_no_LRA[0] + e_bv_at_step_no_LRA[1]
            losses_no_LRA.append(e_bplusv_at_step_no_LRA)
            
        losses = jnp.array(losses)
        losses_no_LRA = jnp.array(losses_no_LRA)
        Ks_shifted = [0] + Ks[:-1]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(Ks, lrs, color='red', label='LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, lrs, color='red', alpha=0.5)
        plt.scatter(Ks, lrs_no_LRA, color='green', label='No LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, lrs_no_LRA, color='green', alpha=0.5)
        # add vertical lines at change points
        for cp in change_point_indices:
            plt.axvline(x=Ks_shifted[cp], color='gray', linestyle='--', alpha=0.5)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step Index')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.subplot(1, 2, 2)
        # use empty circles for Ks, and filled circles for Ks_shifted
        plt.scatter(Ks, losses, color='red', label='LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, losses, color='red', alpha=0.5)
        plt.scatter(Ks, losses_no_LRA, color='green', label='No LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, losses_no_LRA, color='green', alpha=0.5)
        # add vertical lines at change points
        for cp in change_point_indices:
            plt.axvline(x=Ks_shifted[cp], color='gray', linestyle='--', alpha=0.5)
        plt.title('Loss (e_bias + e_var) at Each Step; time: {:.2f} sec'.format(elapsed_time))
        plt.xlabel('Step Index')
        plt.ylabel('Loss')
        # log scale for y axis
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        # save the plot
        plt.savefig("LRA_schedule_and_loss.png")
        print(f"LRA completed in {elapsed_time:.4f} seconds.")
    return e_bv_at_step

nqs = jnp.array([1.1652930752138773,
              0.9293271292641424,
              3.9277975287734592, 
              0.4466304384813954, 
              0.34046502245729804,
              2.280919414389971**2,
              0.9293271292641424])
cfg = Cfg(N=10000, K=50000, B=32, lr=1.0, sch={"decay_at": [0.5, 0.8], "decay_amt": [0.5, 0.5], "B_decay_amt": [1.0, 2.0]})
LRA_interval = 5000
LRA_tol = 0.001

# unit test for _e_bias_var_fast_LRA
def test_e_bias_var_LRA():
    e_bv = _e_bias_var_LRA(nqs, cfg, interval=LRA_interval, LRA_tol=LRA_tol, verbose=True)
    print(f"Test e_bias_var_fast_LRA: e_bv = {e_bv}")


def _e_dim_bv_one_step(nqs, n, bv_factor, step, b_decay_factor = 1.0):
        '''
        Advance the b_factor and v_factor for both e_bias and e_var at dimension n, over one phase in a multi-step schedule
        '''

        #b_factor = 1.0
        #v_factor = 0.0

        p, q, P, Q, e_irr, R, r = nqs
        b_factor, v_factor = bv_factor

        #for step in steps:
        lr = step[0]
        B = step[1]
        num_steps = step[2]
        a = 1.0 - lr * (Q / n**q)

        sumprod_factor, prod_factor = _geom_sum(a**2, num_steps, output_prod = True)
        
        b_factor = b_factor * prod_factor
        # use b_decay_factor to scale the noise at this step
        # a b_decay_factor < 1.0 means reducing the noise (e.g. effect of regularization)
        v_factor = v_factor * prod_factor + lr**2/B * sumprod_factor * b_decay_factor
        bv_factor = jnp.array([b_factor, v_factor])

        return bv_factor

def _get_em_quadrature_points(L, U):
    logL = jnp.log(L)
    logU = jnp.log(U)
    # quadrature points and weights
        # quadrature points and weights
        # Fixed 20-point Gauss-Legendre quadrature for general use
    x = jnp.array([
        -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
        -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
        -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
        -0.0765265211334973, 0.0765265211334973, 0.2277858511416451,
        0.3737060887154195, 0.5108670019508271, 0.6360536807265150,
        0.7463319064601508, 0.8391169718222188, 0.9122344282513259,
        0.9639719272779138, 0.9931285991850949
    ])
    
    w = jnp.array([
        0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
        0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
        0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
        0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
        0.1420961093183820, 0.1316886384491766, 0.1181945319615184,
        0.1019301198172404, 0.0832767415767048, 0.0626720483341091,
        0.0406014298003869, 0.0176140071391521
    ])
    
    # Transform the interval [a, b] to [-1, 1]
    mid = (logU + logL) / 2
    radius = (logU - logL) / 2
    
    # Map points from [-1, 1] to [logL, logU]
    mapped_x = mid + radius * x

    mapped_n = jnp.exp(mapped_x)

    return mapped_n, w


def _init_all_points(L, U, include_1_to_L=True):
    # get quadrature points, then append all integers in 1, 2,..., L
    quad_n, quad_w = _get_em_quadrature_points(L, U)
    if include_1_to_L:
        one_to_L = jnp.arange(1, L+1)
        all_n = jnp.concatenate([quad_n, one_to_L])
        radius = (jnp.log(U) - jnp.log(L)) / 2
        all_w = jnp.concatenate([quad_w * radius, jnp.ones_like(one_to_L)]) 
    else:
        all_n = quad_n
        radius = (jnp.log(U) - jnp.log(L)) / 2
        all_w = quad_w * radius
    # the first twenty points are quadrature points, they get weights quad_w * radius
    # the rest are to be summed directly, they get weight 1.0

    # initialize bv_factors to be 1.0, 0.0
    bv_factors = jnp.tile(jnp.array([1.0, 0.0]), (all_n.shape[0], 1))
    start_factors = jnp.concatenate([all_n[:, None], all_w[:, None], bv_factors], axis=1)
    return start_factors


def _bv_from_bv_factor(nqs, n, bv_factor):
        
        p, q, P, Q, e_irr, R, r = nqs
        b_factor, v_factor = bv_factor
        
        init_R_n = 0.5 * P / n**p
        lambda_n = Q / n**q
        e_bias_n = init_R_n * b_factor
        e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
        return jnp.array([e_bias_n, e_var_n])

# weight norm
def _w2_from_bv_factor(nqs, n, bv_factor):
        
        p, q, P, Q, e_irr, R, r = nqs
        b_factor, v_factor = bv_factor
        

        # in the n-th dimension, the expected weight norm squared is:
        # E[||w_n||^2] = 2 / lambda_n * (E[Bias_n] + E[Var_n]) + (w^*)^2 + 2 * w^* \sqrt(b_factor) * (w_0 - w^*)
        # assume w_0 = 0 for simplicity, this gives
        # E[||w_n||^2] = 2 / lambda_n * (E[Bias_n] + E[Var_n]) + (w^*)^2 - 2 * w^* \sqrt(b_factor) * w^*
        # = 2 / lambda_n * (E[Bias_n] + E[Var_n]) + (1 - 2 * \sqrt(b_factor)) * (w^*)^2
        # and (w^*)^2 = init_R_n / lambda_n

        init_R_n = 0.5 * P / n**p
        lambda_n = Q / n**q
        e_bias_n = init_R_n * b_factor
        e_var_n = 0.5 * lambda_n * (R / n**r) * v_factor
        w_star2_n = P / n**p / (lambda_n)
        #e_w2_n = e_bias_n + e_var_n + (1 - 2 * b_factor) * w_star2_n
        e_w2_n = (1 - 2 * jnp.sqrt(b_factor))  * w_star2_n  + 2 / lambda_n * (e_bias_n + e_var_n)
        #e_w2_n  = w_star2_n  + 2 / lambda_n * (e_bias_n + e_var_n)
        return e_w2_n

@jax.jit
def _em_step(start_factors, nqs, step, verbose= False, b_decay_factor=1.0, return_w2=False):

    # start_gns is the cache of 
    # the set of function values at the quadrature points & the first M points at  
    # the end of the previous step

    # start_gns is an array of shape (num_quad_points, 4)
    # the 4 columns are: n, w, b_facor, v_factor

    quadrature_points= start_factors[:, 0]
    w = start_factors[:, 1]
    bv_factors_curr = start_factors[:, 2:4]

    #b_decay_factor = 1.0 # no decay of noise for now
    bv_factors_stepped = jax.vmap(lambda n, bv_factor: _e_dim_bv_one_step(nqs, n, bv_factor, step, b_decay_factor))(quadrature_points, bv_factors_curr)
    bv_values = jax.vmap(lambda n, bv_factor: _bv_from_bv_factor(nqs, n, bv_factor))(quadrature_points, bv_factors_stepped)
    w2_values = jax.vmap(lambda n, bv_factor: _w2_from_bv_factor(nqs, n, bv_factor))(quadrature_points, bv_factors_stepped)
    
    # Evaluate function at quadrature points
    y_values  = bv_values * jnp.where(quadrature_points[:, None] > L, quadrature_points[:, None], 1.0)
    w2_values  = w2_values * jnp.where(quadrature_points > L, quadrature_points, 1.0)
    # y_values * quadrature_points[:, None] # account for dn = n * dlogn
    # for the non -quadrature points (the integers from 1 to L-1), no need to multiply by n
    
    # Compute integral (vectorized over n)

    # the first twenty points are quadrature points, they get radius and ydn_values weighting (both in w)
    # the rest are to be summed directly, they get weight 1.0
    # (this is already accounted for in w)
    result = jnp.sum(w[:, jnp.newaxis] * y_values, axis=0)
    result_w2 = jnp.sum(w * w2_values, axis=0)

    end_factors = jnp.concatenate([quadrature_points[:, None], w[:, None], bv_factors_stepped], axis=1)

    return result, result_w2, end_factors



@jax.jit
def _em_at(start_factors, nqs, verbose= False, return_w2=False):

    # start_gns is the cache of 
    # the set of function values at the quadrature points & the first M points at  
    # the end of the previous step

    # start_gns is an array of shape (num_quad_points, 4)
    # the 4 columns are: n, w, b_facor, v_factor

    quadrature_points= start_factors[:, 0]
    w = start_factors[:, 1]
    bv_factors_curr = start_factors[:, 2:4]

    bv_values = jax.vmap(lambda n, bv_factor: _bv_from_bv_factor(nqs, n, bv_factor))(quadrature_points, bv_factors_curr)
    w2_values = jax.vmap(lambda n, bv_factor: _w2_from_bv_factor(nqs, n, bv_factor))(quadrature_points, bv_factors_curr)
        

    # Evaluate function at quadrature points
    y_values  = bv_values * jnp.where(quadrature_points[:, None] > L, quadrature_points[:, None], 1.0)
    # y_values * quadrature_points[:, None] # account for dn = n * dlogn
    # for the non -quadrature points (the integers from 1 to L-1), no need to multiply by n

    w2_values  = w2_values * jnp.where(quadrature_points > L, quadrature_points, 1.0)
    
    # Compute integral (vectorized over n)

    # the first twenty points are quadrature points, they get radius and ydn_values weighting (both in w)
    # the rest are to be summed directly, they get weight 1.0
    # (this is already accounted for in w)
    result = jnp.sum(w[:, jnp.newaxis] * y_values, axis=0)
    result_w2 = jnp.sum(w * w2_values, axis=0)

    #end_factors = jnp.concatenate([quadrature_points[:, None], w[:, None], bv_factors_curr], axis=1)

    return result, result_w2 

def _em_by_step(L, U, nqs, steps, include_1_to_L=True,
                verbose=False):
    '''
    EM computation of bias and variance from dimension L+1 to U, over multiple steps
    '''

    time_start = time.time()

    start_factors = _init_all_points(L, U, include_1_to_L=include_1_to_L)
    curr_factors = start_factors

    for step in steps:
        if verbose:
            print(f"EM step: lr={step[0]}, B={step[1]}, num_steps={step[2]}")
        bias_var_step, w2_step, end_factors = _em_step(curr_factors, nqs, step, verbose= verbose)
        curr_factors = end_factors
    
    bias_var = bias_var_step

    time_end = time.time()
    if verbose:
        print(f"Time for EM from {L} to {U} over {len(steps)} steps: {time_end - time_start} seconds")

    return bias_var


# test _em_by_step against 


# make a copy of cfg
cfg_0 = Cfg(N=10000, K=5000, B=32, lr=1.0, sch={"decay_at": [], "decay_amt": [], "B_decay_amt": []})
M = jnp.minimum(jnp.maximum(1, jnp.array((cfg_0.N * 0.05), int)), 100)
L = M
U = cfg_0.N
steps = _process_schedule_steps(cfg_0.lr, cfg_0.B, cfg_0.K, cfg_0.sch) #, interval = 5000)
bv_em_by_step = _em_by_step(L, U, nqs, steps, include_1_to_L=True)
print(f"Test _em_by_step: bv_em_by_step = {bv_em_by_step}")
bv_em_direct = _jit_em_f(L, U, nqs, steps)
bv_reduce_direct = _jit_reduce_f(nqs, L, steps)
print(f"Test _em_by_step: bv_reduce_direct = {bv_reduce_direct}")
print(f"Test _em_by_step: bv_em_direct = {bv_em_direct}")
bv_direct = bv_em_direct + bv_reduce_direct
print(f"Test _em_by_step: bv_direct = {bv_direct}")

#raise ValueError("Stop after test of _em_by_step")


def _e_bias_var_LRA_fast(nqs, cfg, interval = 1000, LRA_tol = 0.05, verbose = False):
    '''
    Fast computation of e_bias and e_var using EM approximation
    (combining the computation of bias and var to save time)
    '''
    N = cfg.N
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)


    K_all = cfg.K
    B = cfg.B
    lr = cfg.lr

    sch = cfg.sch
    steps_all = _process_schedule_steps_LRA(lr, B, K_all, sch, interval = interval)


    K = 0
    curr_lr = cfg.lr

    steps_all_pre_LRA = steps_all

    if verbose:
        # make a plot of the learning rate schedule and the loss (e_bias + e_var) at each step
        import matplotlib.pyplot as plt
        lrs = steps_all[:, 0]
        lrs_no_LRA = steps_all_pre_LRA[:, 0]

        steps_all_no_LRA = _process_schedule_steps_LRA(cfg.lr, cfg.B, cfg.K, cfg.sch, interval = interval)

        Ks = []
        change_point_indices = jnp.where(steps_all[:, 3] == 1)[0]
        losses = []
        losses_no_LRA = []
    
    start_time = time.time()
    # iterate through the rest of the steps
    for i in range(0, steps_all.shape[0]):
        if i == 0:
            # initiate quadrature points for EM
            start_factors = _init_all_points(M, N, include_1_to_L=True)
            if verbose:
                start_factors_no_LRA = _init_all_points(M, N, include_1_to_L=True)
        
        step = steps_all[i, :]
        step_cgpt_ind = step[3]
        step_lr = step[0]
        # update step learning rate with the minimum of current lr and step lr
        step_lr = jnp.minimum(curr_lr, step_lr)
        curr_lr = step_lr
        steps_all = steps_all.at[i, 0].set(step_lr)
        step = steps_all[i, :]

        #e_bv_at_step = _e_bv_step(steps_all[0:i+1, :])
        #e_bplusv_at_step = e_bv_at_step[0] + e_bv_at_step[1]

        # use the _em_step function to advance the quadrature points
        bias_var_step, _, end_factors = _em_step(start_factors, nqs, step, verbose= verbose)
        end_factors = end_factors
        e_bv_at_step = bias_var_step
        e_bplusv_at_step = e_bv_at_step[0] + e_bv_at_step[1]

        # check if last step is a change point
        if step_cgpt_ind == 0.0: # not a change point
            K = K + step[2]
            start_factors = end_factors
            loss = e_bplusv_at_step
            if verbose:
                print (f"Step {i}, not a change point,  lr: {step_lr}, e_bv: {e_bv_at_step}, K: {K}")
        else: # change point
            
            # try to decay learning rate by 0.5
            proposed_lr = curr_lr * 0.5
            steps_all = steps_all.at[i, 0].set(proposed_lr)
            step = steps_all[i, :]
            #e_bv_at_step_proposed = _e_bv_step(steps_all[0:i+1, :])
            #e_bplusv_at_step_proposed = e_bv_at_step_proposed[0] + e_bv_at_step_proposed[1]

            bias_var_step_proposed, _, end_factors_proposed = _em_step(start_factors, nqs, step, verbose= verbose)
            e_bv_at_step_proposed = bias_var_step_proposed
            e_bplusv_at_step_proposed = e_bv_at_step_proposed[0] + e_bv_at_step_proposed[1]
            
            # if proposed loss is lower than the current loss, by a tolerance, accept the change
            decay_count = 0
            loss_reduction = e_bplusv_at_step - e_bplusv_at_step_proposed
            loss_reduction = jnp.round(loss_reduction, decimals=5)

            loss = e_bplusv_at_step

            while loss_reduction > LRA_tol and decay_count < 10:
                
                curr_lr = proposed_lr
                end_factors = end_factors_proposed
                loss = e_bplusv_at_step_proposed
                print("Step", i, "accepted lr decay to", curr_lr, "new loss:", loss)

                e_bv_at_step = e_bv_at_step_proposed
                e_bplusv_at_step = e_bplusv_at_step_proposed
                
                decay_count += 1

                if verbose:
                    print (f"Step {i}, change point, accepted decay, loss_reduction: {loss_reduction},  lr: {curr_lr}, e_bv: {e_bv_at_step}, K: {K}")
                            # try to decay learning rate by 0.5
                
                proposed_lr = curr_lr * 0.5
                steps_all = steps_all.at[i, 0].set(proposed_lr)
                step = steps_all[i, :]

                #e_bv_at_step_proposed = _e_bv_step(steps_all[0:i+1, :])
                #e_bplusv_at_step_proposed = e_bv_at_step_proposed[0] + e_bv_at_step_proposed[1]
                bias_var_step_proposed, _, end_factors_proposed = _em_step(start_factors, nqs, step, verbose= verbose)
                e_bv_at_step_proposed = bias_var_step_proposed
                e_bplusv_at_step_proposed = e_bv_at_step_proposed[0] + e_bv_at_step_proposed[1]
                loss_reduction = e_bplusv_at_step - e_bplusv_at_step_proposed
                loss_reduction = jnp.round(loss_reduction, decimals=5)

            steps_all = steps_all.at[i, 0].set(curr_lr)
            K = K + step[2]
            start_factors = end_factors
            loss = loss
            

        if verbose:
            Ks.append(K)
            losses.append(loss)

            # for comparison, compute the loss without LRA
            step_no_LRA = steps_all_no_LRA[i, :]
            e_bv_at_step_no_LRA, _, end_factors_no_LRA = _em_step(start_factors_no_LRA, nqs, step_no_LRA, verbose= verbose)
            e_bplusv_at_step_no_LRA = e_bv_at_step_no_LRA[0] + e_bv_at_step_no_LRA[1]
            start_factors_no_LRA = end_factors_no_LRA
            loss_no_LRA = e_bplusv_at_step_no_LRA

            losses_no_LRA.append(loss_no_LRA)

    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        # make a plot of the learning rate schedule and the loss (e_bias + e_var) at each step
        import matplotlib.pyplot as plt
        lrs = steps_all[:, 0]
        lrs_no_LRA = steps_all_no_LRA[:, 0]

            
        losses = jnp.array(losses)
        losses_no_LRA = jnp.array(losses_no_LRA)
        Ks_shifted = [0] + Ks[:-1]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(Ks, lrs, color='red', label='LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, lrs, color='red', alpha=0.5)
        plt.scatter(Ks, lrs_no_LRA, color='green', label='No LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, lrs_no_LRA, color='green', alpha=0.5)
        # add vertical lines at change points
        for cp in change_point_indices:
            plt.axvline(x=Ks_shifted[cp], color='gray', linestyle='--', alpha=0.5)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step Index')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.subplot(1, 2, 2)
        # use empty circles for Ks, and filled circles for Ks_shifted
        plt.scatter(Ks, losses, color='red', label='LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, losses, color='red', alpha=0.5)
        plt.scatter(Ks, losses_no_LRA, color='green', label='No LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, losses_no_LRA, color='green', alpha=0.5)
        # add vertical lines at change points
        for cp in change_point_indices:
            plt.axvline(x=Ks_shifted[cp], color='gray', linestyle='--', alpha=0.5)

        plt.title('Loss (e_bias + e_var) at Each Step; time: {:.2f} sec'.format(elapsed_time))
        plt.xlabel('Step Index')
        plt.ylabel('Loss')
        # log scale for y axis
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        # save the plot
        plt.savefig("LRA_schedule_and_loss_fast.png")
        if verbose:
            print(f"LRA completed in {elapsed_time:.4f} seconds.")
        
    return e_bv_at_step


def _e_bias_var_SN_fast(nqs, cfg, interval = 1000, LRA_tol = None, verbose = False):
    '''
    Fast computation of e_bias and e_var using EM approximation
    (combining the computation of bias and var to save time)
    '''
    N = cfg.N
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)


    K_all = cfg.K
    B = cfg.B
    lr = cfg.lr

    sch = cfg.sch
    steps_all = _process_schedule_steps_LRA(lr, B, K_all, sch, interval = interval)


    K = 0
    init_lr = cfg.lr
    curr_lr = cfg.lr

    steps_all_pre_LRA = steps_all

    if verbose:
        # make a plot of the learning rate schedule and the loss (e_bias + e_var) at each step
        import matplotlib.pyplot as plt
        lrs = steps_all[:, 0]
        lrs_no_LRA = steps_all_pre_LRA[:, 0]

        steps_all_no_LRA = _process_schedule_steps_LRA(cfg.lr, cfg.B, cfg.K, cfg.sch, interval = interval)

        Ks = []
        change_point_indices = jnp.where(steps_all[:, 3] == 1)[0]
        losses = []
        losses_no_LRA = []
        w2s = []
        w2s_no_LRA = []
        lr_scales = []
    
    start_time = time.time()
    # iterate through the rest of the steps
    for i in range(0, steps_all.shape[0]):
        if i == 0:
            # initiate quadrature points for EM
            start_factors = _init_all_points(M, N, include_1_to_L=True)
            if verbose:
                start_factors_no_LRA = _init_all_points(M, N, include_1_to_L=True)
            # initiate the noise scale
            # start with noise scale 1.0
            # compute loss
            bias_var_init, w2_init = _em_at(start_factors, nqs, verbose= verbose, return_w2=True)
            loss_init = bias_var_init[0] + bias_var_init[1]
            loss = loss_init
            w2 = w2_init

        
        noise_scale = 1.0 #loss/loss_init
        if w2 < 0.0: # prev candidate 1.0
            lr_scale = 1.0 # this is to avoid numerical issues when w2 is very small; no impact in theory
        else:
            #hidden_width = jnp.exp(-0.0571 + 0.3687 * jnp.log(N)) 
            #raise ValueError("hidden_width: ", hidden_width)
            #lr_scale = jnp.sqrt(1e4/w2)
            #jnp.sqrt(hidden_width) / jnp.sqrt(w2) * 6
            #lr_scale = jnp.sqrt(30 * hidden_width / w2)
            #raise ValueError("N: {}, lr_scale_0: {}, lr_scale_1: {}, w2: {}, hidden_width: {}".format(N, lr_scale_0, lr_scale_1, w2, hidden_width))
            #jnp.sqrt(30*hidden_width/w2)
            #raise ValueError(30 * hidden_width)
            #lr_scale = jnp.sqrt(630*N**0.6/1e3/w2)
            #lr_scale = jnp.sqrt(N/5e3/w2) #Prev Candidate
            lr_scale = 1 - w2/(N * 0.02**2 + w2)
            #lr_scale = 1 - w2/(N * 0.02**2*3 + w2)

            #lr_scale = jnp.sqrt(N)/w2 ## at end of training appx. 0.2 for N = 1e7 and w2 = 5e4
            #lr_scale = jnp.sqrt(2e3/w2)
        # compute lr according to cosine schedule
        # first compute which step we are in
        iteration = jnp.sum(steps_all[0:i, 2]) + steps_all[i, 2]/2
        total_iterations = cfg.K
        # use the cosine_lr_with_warmup function to compute the lr scale
        #cosine_lr_scale = _cosine_lr_with_warmup(k = iteration, 
        #                                         num_training_steps= total_iterations)
        
        cosine_lr_scale = 1.0
        #raise ValueError("cosine_lr_scale: {}, lr_scale: {}, iteration: {}, total_iterations: {}".format(cosine_lr_scale, lr_scale, iteration, total_iterations))

        if verbose:
                print(f"Step {i}, updated noise scale to {noise_scale} based on loss {loss} and initial loss {loss_init}")
                print(f"Step {i}, updated lr scale to {lr_scale} based on w2 {w2} and target 1e4")
                lr_scales.append(lr_scale)
                print(f"Step {i}, lr scales: {lr_scales}")
        step = steps_all[i, :]
        step_cgpt_ind = step[3]
        step_lr = step[0]
        # update step learning rate with the minimum of current lr and step lr
        #step_lr = jnp.maximum(1e-10, jnp.minimum(step_lr * cosine_lr_scale, step_lr * lr_scale)) #Prev Candidate
        step_lr =jnp.minimum(curr_lr, jnp.minimum(step_lr, init_lr * lr_scale))
        curr_lr = step_lr
        steps_all = steps_all.at[i, 0].set(step_lr)
        step = steps_all[i, :]

        #e_bv_at_step = _e_bv_step(steps_all[0:i+1, :])
        #e_bplusv_at_step = e_bv_at_step[0] + e_bv_at_step[1]

        # use the _em_step function to advance the quadrature points
        bias_var_step, w2_step, end_factors = _em_step(start_factors, nqs, step, 
                                              b_decay_factor= noise_scale, 
                                              verbose= verbose, return_w2=True)
        end_factors = end_factors
        e_bv_at_step = bias_var_step
        e_bplusv_at_step = e_bv_at_step[0] + e_bv_at_step[1]
        

        K = K + step[2]
        start_factors = end_factors
        loss = e_bplusv_at_step
        w2 = w2_step

        if verbose:
            print (f"Step {i}, not a change point,  lr: {step_lr}, e_bv: {e_bv_at_step}, K: {K}")

        if verbose:
            Ks.append(K)
            losses.append(loss)
            w2s.append(w2)

            # for comparison, compute the loss without LRA
            step_no_LRA = steps_all_no_LRA[i, :]
            e_bv_at_step_no_LRA, w2_no_LRA,end_factors_no_LRA = _em_step(start_factors_no_LRA, nqs, step_no_LRA, verbose= verbose)
            e_bplusv_at_step_no_LRA = e_bv_at_step_no_LRA[0] + e_bv_at_step_no_LRA[1]
            start_factors_no_LRA = end_factors_no_LRA
            loss_no_LRA = e_bplusv_at_step_no_LRA

            losses_no_LRA.append(loss_no_LRA)
            w2s_no_LRA.append(w2_no_LRA)

    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        # make a plot of the learning rate schedule and the loss (e_bias + e_var) at each step
        import matplotlib.pyplot as plt
        lrs = steps_all[:, 0]
        lrs_no_LRA = steps_all_no_LRA[:, 0]

            
        losses = jnp.array(losses)
        losses_no_LRA = jnp.array(losses_no_LRA)
        Ks_shifted = [0] + Ks[:-1]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.scatter(Ks, lrs, color='red', label='LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, lrs, color='red', alpha=0.5)
        plt.scatter(Ks, lrs_no_LRA, color='green', label='No LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, lrs_no_LRA, color='green', alpha=0.5)
        # add vertical lines at change points
        for cp in change_point_indices:
            plt.axvline(x=Ks_shifted[cp], color='gray', linestyle='--', alpha=0.5)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step Index')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.subplot(1, 3, 2)
        # use empty circles for Ks, and filled circles for Ks_shifted
        plt.scatter(Ks, losses, color='red', label='LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, losses, color='red', alpha=0.5)
        plt.scatter(Ks, losses_no_LRA, color='green', label='No LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, losses_no_LRA, color='green', alpha=0.5)
        # add vertical lines at change points
        for cp in change_point_indices:
            plt.axvline(x=Ks_shifted[cp], color='gray', linestyle='--', alpha=0.5)

        plt.title('Loss (e_bias + e_var) at Each Step; time: {:.2f} sec'.format(elapsed_time))
        plt.xlabel('Step Index')
        plt.ylabel('Loss')
        # log scale for y axis
        plt.yscale('log')
        plt.legend()

        plt.subplot(1, 3, 3)
        # use empty circles for Ks, and filled circles for Ks_shifted
        plt.scatter(Ks, w2s, color='red', label='LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, w2s, color='red', alpha=0.5)
        plt.scatter(Ks, w2s_no_LRA, color='green', label='No LRA', alpha=0.5, facecolors='none')
        plt.scatter(Ks_shifted, w2s_no_LRA, color='green', alpha=0.5)
        # add vertical lines at change points
        for cp in change_point_indices:
            plt.axvline(x=Ks_shifted[cp], color='gray', linestyle='--', alpha=0.5)

        plt.title('Final w2 is {:.4f}'.format(w2))
        plt.ylabel('Weight Norm (w2)')
        # log scale for y axis
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        # save the plot
        plt.savefig("LRA_schedule_and_loss_fast_SN.png")
        if verbose:
            print(f"LRA completed in {elapsed_time:.4f} seconds.")
        
    return e_bv_at_step, w2


def _e_bias_var_reg_fast(nqs, cfg, interval = 1000, LRA_tol = None, verbose = False):
    '''
    Fast computation of e_bias and e_var using EM approximation
    (combining the computation of bias and var to save time)
    '''
    N = cfg.N
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)

    K_all = cfg.K
    B = cfg.B
    lr = cfg.lr
    sch = cfg.sch
    steps_all = _process_schedule_steps_LRA(lr, B, K_all, sch, interval = interval)

    K = 0
    init_lr = cfg.lr
    curr_lr = cfg.lr
    steps_all_pre_LRA = steps_all
    
    start_time = time.time()
    # iterate through the rest of the steps
    for i in range(0, steps_all.shape[0]):
        if i == 0:
            # initiate quadrature points for EM
            start_factors = _init_all_points(M, N, include_1_to_L=True)
            if verbose:
                start_factors_no_LRA = _init_all_points(M, N, include_1_to_L=True)
            # initiate the noise scale
            # start with noise scale 1.0
            # compute loss
            bias_var_init, w2_init = _em_at(start_factors, nqs, verbose= verbose, return_w2=True)
            loss_init = bias_var_init[0] + bias_var_init[1]
            loss = loss_init
            w2 = w2_init

        noise_scale = 1.0 #loss/loss_init
        lr_scale = 1 - w2/(N * 0.02**2 + w2)

        step = steps_all[i, :]
        step_lr = step[0]
        # update step learning rate with the minimum of current lr and step lr
        #step_lr = jnp.maximum(1e-8, jnp.minimum(step_lr * cosine_lr_scale, step_lr * lr_scale))
        step_lr = init_lr * lr_scale
        #jnp.minimum(curr_lr, jnp.minimum(step_lr, init_lr * lr_scale))
        steps_all = steps_all.at[i, 0].set(step_lr)
        step = steps_all[i, :]

        #e_bv_at_step = _e_bv_step(steps_all[0:i+1, :])
        #e_bplusv_at_step = e_bv_at_step[0] + e_bv_at_step[1]

        # use the _em_step function to advance the quadrature points
        bias_var_step, w2_step, end_factors = _em_step(start_factors, nqs, step, 
                                              b_decay_factor= noise_scale, 
                                              verbose= verbose, return_w2=True)
        end_factors = end_factors
        e_bv_at_step = bias_var_step
        e_bplusv_at_step = e_bv_at_step[0] + e_bv_at_step[1]
        

        K = K + step[2]
        start_factors = end_factors
        loss = e_bplusv_at_step
        w2 = w2_step

        
    return e_bv_at_step, w2


nqs = jnp.array([1.1652930752138773,
              0.9293271292641424,
              3.9277975287734592, 
              0.4466304384813954, 
              0.34046502245729804,
              2.280919414389971**2,
              0.9293271292641424])

#p: 1.1097530321691766
#q: 0.5933368010361162
##P: 3.137653267064672
#Q: 0.9591955803653882
#e_irr: 0.3314231265816201
#R: 2.874398089570365
#r: 1.4903324960432542

nqs = jnp.array([1.1097530321691766,
              0.5933368010361162,
              3.137653267064672, 
              0.9591955803653882, 
              0.3314231265816201,
              2.874398089570365,
              1.4903324960432542
              ])
cfg = Cfg(#N = 4000000, K=1171, B = 234, lr = 2.0,
    N=128*1e6, K=40000, B=48, lr=2.0, 
          #sch={"decay_at": [0.5, 0.8], "decay_amt": [0.5, 0.5], "B_decay_amt": [1.0, 2.0]}
          sch = {"decay_at": [0.5], "decay_amt": [1.0], "B_decay_amt": [1.0]}
          )
LRA_interval = 500

# compute e_appx and e_irr for reference
e_appx = _e_appx(nqs, cfg)
e_irr = _e_irr(nqs, cfg)


# unit test for _e_bias_var_rdnoise_fast

def test_e_bias_var_SN_fast():
    print("Testing _e_bias_var_rdnoise_fast...")
    print("nqs:", nqs)
    print("cfg:", cfg)
    print("LRA_interval:", LRA_interval)
    e_bv, w2 = _e_bias_var_SN_fast(nqs, cfg, interval=LRA_interval, verbose=True)
    #e_bv_LRA = _e_bias_var_LRA_fast(nqs, cfg, interval=LRA_interval, LRA_tol=0.05, verbose=True)
    # compare with vanilla e_bias_var
    e_bv_vanilla = _e_bias_var_fast(nqs, cfg)
   # print(f"Test e_bias_var_SN_fast vs LRA: e_bv = {e_bv}, e_bv_LRA = {e_bv_LRA}")
   # print(f"Total: LRA: = {e_bv_LRA[0] + e_bv_LRA[1] + e_appx + e_irr}")
    print(f"Total: SN fast = {e_bv[0] + e_bv[1] + e_appx + e_irr}")
    print(f"Total: Vanilla = {e_bv_vanilla[0] + e_bv_vanilla[1] + e_appx + e_irr}")
    raise ValueError("Stop after test of _e_bias_var_SN_fast")

#test_e_bias_var_SN_fast()

def _risk_LRA(nqs, cfg, LRA_tol = 0.05):
    K_target = cfg.K
    if K_target > 10:
        stair_width = max(K_target //100, min(1000, K_target // 10))
        print(f"_risk_LRA: stair_width = {stair_width}")
    else:
        stair_width = 1


    # Regularization Method Switch 
    #e_est_bv = _e_bias_var_LRA_fast(nqs, cfg, interval = stair_width, LRA_tol = LRA_tol, verbose = False)
    e_est_bv,_ = _e_bias_var_SN_fast(nqs, cfg, interval = stair_width, LRA_tol = LRA_tol, verbose = False)
    print(f"_risk_LRA: e_est_bv = {e_est_bv}")
    e_appx = _e_appx(nqs, cfg)
    print(f"_risk_LRA: e_appx = {e_appx}")
    e_irr = _e_irr(nqs, cfg)
    print(f"_risk_LRA: e_irr = {e_irr}")
    risk = e_est_bv[0] + e_est_bv[1] + e_appx + e_irr
    return risk

# test _risk_LRA
#def test_risk_LRA():
#    risk = _risk_LRA(nqs, cfg, LRA_tol = LRA_tol)
#    print(f"Test _risk_LRA: risk = {risk}")
#test_risk_LRA()

# ---------------------------------------------------- #
#        Wrappers for use in the paper pipeline        #
# ---------------------------------------------------- #


# --- Computing
def EM_nqs_from_cfg_six_standard(nqs_cfg, working_file_path = None):

    p = nqs_cfg.p
    q = nqs_cfg.q
    P = nqs_cfg.P
    Q = nqs_cfg.Q
    e_irr = nqs_cfg.e_irr
    R = nqs_cfg.R
    r = nqs_cfg.r

    nqs = jnp.array([p, q, P, Q, e_irr, R, r])  # the parameters are p, q, P, Q, e_irr, R, r

    h = nqs_cfg.h
    init_lr = h.lr 
    beta = h.momentum
    B = h.B
    K = h.K
    N = h.N
    step_decay_schedule = h.step_decay_schedule

    if beta != 0.0:
        raise NotImplementedError("momentum not implemented in fast nqs")
    
    
    if h.lr_schedule == "step":
        sch = step_decay_schedule
    elif h.lr_schedule in ["constant"]:
        sch = {"decay_at":[], "decay_amt":[], "B_decay_amt":[]}
    else:
        raise ValueError("for fast nqs, lr_schedule must be one of 'step' or 'constant'")

    iters_to_calc = [K]
  
    nqs_risks = []
    for k in iters_to_calc:
            nqs_risk = _risk(nqs, Cfg(N=N, K=k, B=B, lr=init_lr, sch=sch))
            nqs_risks.append(nqs_risk)

    nqs_risks = jnp.array(nqs_risks)
    nqs_df = pd.DataFrame({"nqs_iter": iters_to_calc, 
                           "nqs_risk": nqs_risks})
    
    return {"nqs_df": nqs_df}


# --- Computing
def EM_nqs_from_cfg_six_optimized(nqs_cfg, working_file_path = None, LRA_tol=0.05):
    
    p = nqs_cfg.p
    q = nqs_cfg.q
    P = nqs_cfg.P
    Q = nqs_cfg.Q
    e_irr = nqs_cfg.e_irr
    R = nqs_cfg.R
    r = nqs_cfg.r
    nqs = jnp.array([p, q, P, Q, e_irr, R, r])  # the parameters are p, q, P, Q, e_irr, R, r

    h = nqs_cfg.h
    init_lr = h.lr 
    beta = h.momentum
    B = h.B
    K = h.K
    N = h.N
    step_decay_schedule = h.step_decay_schedule

    if beta != 0.0:
        raise NotImplementedError("momentum not implemented in fast nqs")
    
    
    if h.lr_schedule in ["optimized","cosine"]:
        #raise ValueError(step_decay_schedule)
        # check if sch is a dictionary
        if step_decay_schedule == "na"  : # use constant schedule
            sch = {"decay_at": [], "decay_amt": [], "B_decay_amt": []}
        # convert OmegaConf to dictionary
        else:
            sch = {"decay_at": step_decay_schedule.decay_at,
                   "decay_amt": step_decay_schedule.decay_amt,
                   "B_decay_amt": step_decay_schedule.B_decay_amt}
            #raise ValueError(sch)
    else:
        raise ValueError("to use fast nqs with LRA adapt, requested lr_schedule must be 'optimized'")

    iters_to_calc = [K]
  
    nqs_risks = []
    for k in iters_to_calc:
            nqs_risk = _risk_LRA(nqs, Cfg(N=N, K=k, B=B, lr=init_lr, sch=sch), LRA_tol=LRA_tol)
            nqs_risks.append(nqs_risk)

    nqs_risks = jnp.array(nqs_risks)
    nqs_df = pd.DataFrame({"nqs_iter": iters_to_calc, 
                           "nqs_risk": nqs_risks})
    
    
    return {"nqs_df": nqs_df}



# --- Computing
def EM_nqs_from_cfg_six_projected(nqs_cfg, working_file_path = None, LRA_tol=0.05):
    
    p = nqs_cfg.p
    q = nqs_cfg.q
    P = nqs_cfg.P
    Q = nqs_cfg.Q
    e_irr = nqs_cfg.e_irr
    R = nqs_cfg.R
    r = nqs_cfg.r
    nqs = jnp.array([p, q, P, Q, e_irr, R, r])  # the parameters are p, q, P, Q, e_irr, R, r

    h = nqs_cfg.h
    init_lr = h.lr 
    beta = h.momentum
    B = h.B
    K = h.K
    N = h.N
    step_decay_schedule = h.step_decay_schedule

    if beta != 0.0:
        raise NotImplementedError("momentum not implemented in fast nqs")
    
    
    if h.lr_schedule == "optimized":
        # check if sch is a dictionary
        if isinstance(step_decay_schedule, dict):
            sch = step_decay_schedule
        else: # use constant schedule
            sch = {"decay_at": [], "decay_amt": [], "B_decay_amt": []}
    else:
        raise ValueError("to use fast nqs with LRA adapt, requested lr_schedule must be 'optimized'")

    iters_to_calc = [K]
  
    nqs_risks = []
    for k in iters_to_calc:
            nqs_risk = _risk_LRA(nqs, Cfg(N=N, K=k, B=B, lr=init_lr, sch=sch), LRA_tol=LRA_tol)
            nqs_risks.append(nqs_risk)

    nqs_risks = jnp.array(nqs_risks)
    nqs_df = pd.DataFrame({"nqs_iter": iters_to_calc, 
                           "nqs_risk": nqs_risks})
    
    
    return {"nqs_df": nqs_df}



# ---- Fitting 
def fit_nqs(h_dicts, nn_losses, seed, number_of_initializations, param_ranges_raw,
            gtol, max_steps):


    list_of_nqs_inits = latin_hypercube_initializations(seed = seed,
        num_inits = number_of_initializations,
        param_names = ['p', 'q', 'P', 'Q', 'e_irr', 'R', 'r'],
        param_ranges = {
            'p': (param_ranges_raw['a'][0], param_ranges_raw['a'][1]),
            'q': (param_ranges_raw['b'][0], param_ranges_raw['b'][1]),
            'P': (param_ranges_raw['ma'][0], param_ranges_raw['ma'][1]),
            'Q': (param_ranges_raw['mb'][0], param_ranges_raw['mb'][1]),
            'e_irr': (param_ranges_raw['c'][0], param_ranges_raw['c'][1]),
            'R': (param_ranges_raw['sigma'][0]**2, param_ranges_raw['sigma'][1]**2),
            'r': (param_ranges_raw['b'][0], param_ranges_raw['b'][1])
        } ,
        r_equals_q = False)
    # h_dicts is a list of dictionaries, each dictionary is a hyperparameter dictionary
    # ls is a list of losses, each loss is a number

    # convert the list of dictionaries into a list of (N,K,B, folds) tuples, by
    # traverse the list of dictionaries
    cfg_arrs = []
    for h_dict in h_dicts:
        N = h_dict["N"]
        K = h_dict["K"]
        B = h_dict["B"]
        lr = h_dict["lr"]

        if h_dict["lr_schedule"] == "step":
            #step_decay_schedule = h_dict["step_decay_schedule"] #json.loads(h_dict["step_decay_schedule"])
            #no_sch = False
            raise NotImplementedError("step lr_schedule not implemented in fast nqs fitting")
        elif h_dict["lr_schedule"] in ["constant","cosine"]: # Treat cosine as constant in fitting
            pass
        else:
            raise ValueError("for fast nqs, lr_schedule must be one of 'step' or 'constant/cosine'")

        cfg_arrs.append(jnp.array([N, K, B, lr]))
    
    ys = jnp.array(nn_losses)
    cfgs = jnp.array(cfg_arrs)

    start_time = time.time()
    best_nqs, best_loss, best_idx, trajectories = _fit_nqs(list_of_nqs_inits, 
                                                 cfgs, 
                                                 ys, 
                                                 itrs=max_steps,
                                                 return_trajectories=True,
                                                 tie_r_and_q= False)
    end_time = time.time()

    print(f"_fit_nqs computation time: {end_time - start_time} seconds")
    print(f"Best NQS found: {best_nqs}")
    print(f"Best loss found: {best_loss}")

    # save a log file with the computation time and best nqs found
    with open("fit_nqs_log.txt", "w") as f:
        f.write(f"_fit_nqs computation time: {end_time - start_time} seconds\n")
        f.write(f"Best NQS found: {best_nqs}\n")
        f.write(f"Best loss found: {best_loss}\n")
        f.write(f"Best loss index: {best_idx}\n")

    fit_metric_value = jnp.array(best_loss)

    # convert the fitted nqs into a dictionary
    fitted_nqs_dict = {'p': best_nqs[0],
                       'q': best_nqs[1],
                       'P': best_nqs[2],
                       'Q': best_nqs[3],
                       'e_irr': best_nqs[4],
                       'R': best_nqs[5],
                       'r': best_nqs[6]}
    # convert the fitted_nqs_dict values into floats
    fitted_nqs_dict = {k: float(v) for k, v in fitted_nqs_dict.items()}

    eval_metric_value = jnp.sqrt(2 * fit_metric_value) * 100

    return fitted_nqs_dict, eval_metric_value, None #trajectories_formatted


if __name__ == "__main__":
    # sgd
    #p: 1.1429154522145861
    #q: 0.6954562288109772
    #P: 4.921070379165702
    #Q: 0.6940112063285089
    #e_irr: 1.1193503050399043
    #R: 4.548347442964086
    #r: 2.3212863962486985
    # test e_est


    # alternatively, adam
    # p: 1.1093095149545336
    # q: 0.5519423605927287
    # P: 2.947890136142957
    # Q: 0.9597077951349916
    # e_irr: 0.5129314985128063
    # R: 5.537452972513861
    # r: 1.4060703012999352

    nqs_array = jnp.array([1.1093095149545336, 
                        0.5519423605927287, 
                        2.947890136142957, 
                        0.9597077951349916, 
                        0.5129314985128063, 
                        5.537452972513861, 
                        1.4060703012999352])
    
    config_obj = Cfg(N=10000000, B=48, K=2500, lr=1.999, sch={"decay_at":[], "decay_amt":[], "B_decay_amt":[]})

    test_risk_flag = True
    if test_risk_flag:
        risk = _risk(nqs_array, config_obj)
        print(f"Test risk is {risk}")

    test_simulation_flag = True
    if test_simulation_flag:

        n_sims = 3
        test_outts = []
        kkey = jax.random.PRNGKey(0)
        norm_bound = jnp.inf
        for sim in range(n_sims):
            print(f"Starting simulation {sim+1} / {n_sims}")
            # split 
            kkey_sim, kkey = jax.random.split(kkey)
            test_outt_sim = _e_est_simulate(nqs_array, 
                                    config_obj,
                                    kkey_sim,
                                    max_w_norm=norm_bound,
                                    save_traj=True)
            test_outts.append(test_outt_sim)
        
        print("Completed all simulations.")
        # compute average of test_outts
        test_outt = (jnp.mean(jnp.array([outt[0] for outt in test_outts])),
                    jnp.mean(jnp.array([outt[1] for outt in test_outts]), axis=0),
                    jnp.mean(jnp.array([outt[2] for outt in test_outts]), axis=0))

        _plot_traj(test_outt[1], test_outt[2], K=len(test_outt[1]))
        print(f"Test e_est is {test_outt[0]}")
        print(f"The final norm is {test_outt[2][-1]}")

        print("No more tests to run in main for fast_nqs.py")
