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



def get_no_reparam_funcs():
    def nqs_to_y(nqs):
        # reparametrize nqs
        a = nqs[0]
        b = nqs[1]
        ma = nqs[2]
        mb = nqs[3]
        c = nqs[4]
        sigma = nqs[5]

        return jnp.array([a, b, ma, mb, c, sigma])
    
    def y_to_ynorm(y, norm_const = 1e5):
        # reparametrize nqs
        a = y[0]
        b = y[1]
        ma = jnp.log(y[2])/jnp.log(norm_const)
        mb = jnp.log(y[3])/jnp.log(norm_const)    
        c = jnp.log(y[4])/jnp.log(norm_const)
        sigma = jnp.log(y[5])/jnp.log(norm_const)

        return jnp.array([a, b, ma, mb, c, sigma])

    def ynorm_to_y(y, norm_const = 1e5):
        # reparametrize nqs
        a = y[0]
        b = y[1]
        ma = jnp.exp(y[2]*jnp.log(norm_const))
        mb = jnp.exp(y[3]*jnp.log(norm_const))
        c = jnp.exp(y[4]*jnp.log(norm_const))
        sigma = jnp.exp(y[5]*jnp.log(norm_const))

        return jnp.array([a, b, ma, mb, c, sigma])
    
    def y_to_nqs(y):
        # reparametrize nqs
        a = y[0]
        b = y[1]
        ma = y[2]
        mb = y[3]
        c = y[4]
        sigma = y[5]

        return jnp.array([a, b, ma, mb, c, sigma])
    #NQS(a=a, b=b, ma=ma, mb=mb, c=c, sigma=sigma)
    
    def jac_nqs_to_jac_y(jac_nqs, nqs):
        # reparametrize nqs
        da = jac_nqs[0]
        db = jac_nqs[1]
        dma = jac_nqs[2]
        dmb = jac_nqs[3]
        dc = jac_nqs[4]
        dsigma = jac_nqs[5]

        return jnp.array([da, db, dma, dmb, dc, dsigma])
    
    def jac_y_to_jac_ynorm(jac_y, y, norm_const = 1e5):
        da = jac_y[0]
        db = jac_y[1]
        dma = jac_y[2] * y[2] * jnp.log(norm_const)
        dmb = jac_y[3] * y[3] * jnp.log(norm_const)
        dc = jac_y[4] * y[4] * jnp.log(norm_const)
        dsigma = jac_y[5] * y[5] * jnp.log(norm_const)

        return jnp.array([da, db, dma, dmb, dc, dsigma])

    def jac_ynorm_to_jac_y(jac_ynorm, ynorm, norm_const = 1e5):
        da = jac_ynorm[0]
        db = jac_ynorm[1]
        dma = jac_ynorm[2] / (norm_const ** ynorm[2]) / jnp.log(norm_const)
        dmb = jac_ynorm[3] / (norm_const ** ynorm[3]) / jnp.log(norm_const)
        dc = jac_ynorm[4] / (norm_const ** ynorm[4]) / jnp.log(norm_const)
        dsigma = jac_ynorm[5] / (norm_const ** ynorm[5]) / jnp.log(norm_const)

        return jnp.array([da, db, dma, dmb, dc, dsigma])

    def jac_y_to_jac_nqs(jac_y, y):
        da = jac_y[0]
        db = jac_y[1]
        dma = jac_y[2]
        dmb = jac_y[3]
        dc = jac_y[4]
        dsigma = jac_y[5]

        return  jnp.array([da, db, dma, dmb, dc, dsigma])
    #NQS(a=da, b=db, ma=dma, mb=dmb, c=dc, sigma=dsigma)

    def nqs_to_params_dict(nqs):
      #  dictt = {"a": nqs.a, "b": nqs.b, "ma": nqs.ma, 
       #         "mb": nqs.mb, "c": nqs.c, "sigma": nqs.sigma}
        dictt = {"a": nqs[0], "b": nqs[1], "ma": nqs[2], 
                "mb": nqs[3], "c": nqs[4], "sigma": nqs[5]}
        return dictt
    
    return nqs_to_y, y_to_ynorm, ynorm_to_y, y_to_nqs, jac_nqs_to_jac_y, jac_y_to_jac_ynorm, jac_ynorm_to_jac_y, jac_y_to_jac_nqs, nqs_to_params_dict
    

nqs_to_y, y_to_ynorm, ynorm_to_y, y_to_nqs, jac_nqs_to_jac_y, jac_y_to_jac_ynorm, jac_ynorm_to_jac_y, jac_y_to_jac_nqs, nqs_to_params_dict = get_no_reparam_funcs()
    
def _to_x(nqs):
    return y_to_ynorm(nqs_to_y(nqs))

def _to_nqs(x):
    return y_to_nqs(ynorm_to_y(x))


#############################################
# Methods                                   #
#############################################

def get_Q(n, nqs, M = 1):
    """Compute the quadratic coefficient Q(n) for dimension n.
    
    Args:
        n (int): Dimension index
        nqs (NQS): System parameters
    
    Returns:
        float: Q(n) = mb * n^(-b)
    """
    # jnp clip n at M
    n_clipped = jnp.clip(n, M, None)
    b = nqs[1]
    mb = nqs[3]
    return mb * n_clipped ** (-b)
            #nqs.mb * n_clipped ** (-nqs.b)

def get_xstar(n, nqs, M = 1):
    """Compute the optimal value x*(n) for dimension n.
    
    Args:
        n (int): Dimension index
        nqs (NQS): System parameters
    
    Returns:
        float: x*(n) = sqrt(ma * n^(b-a) / mb)
    """
    # jnp clip n at M
    n_clipped = jnp.clip(n, M, None)
    a = nqs[0]
    b = nqs[1]
    ma = nqs[2]
    mb = nqs[3]

    return jnp.sqrt(ma * n_clipped ** (b-a) / mb)
#jnp.sqrt(nqs.ma * n_clipped ** (nqs.b-nqs.a) / nqs.mb)

def get_V(n, B, nqs, M = 1):
    """Compute the noise variance V(n) for dimension n.
    
    Args:
        n (int): Dimension index
        B (int): Mini-batch size
        nqs (NQS): System parameters
    
    Returns:
        float: V(n) = sigma^2 * n^(-b) / B
    """
    # jnp clip n at M
    n_clipped = jnp.clip(n, M, None)
    sigma = nqs[5]
    b = nqs[1]
    return sigma ** 2 * n_clipped ** (-b) / B
#nqs.sigma ** 2 * n_clipped ** (-nqs.b) / B

def get_T(n, nqs, A, B_mat, C, factor = 1):
    """Compute the effective transition matrix T(n) for dimension n.
    
    Args:
        n (int): Dimension index
        nqs (NQS): System parameters
        mats (mats): System matrices
        factor (float): Learning rate factor
    
    Returns:
        ndarray: T(n) = A + Q(n) * factor * B @ C
    """

    return A + get_Q(n, nqs) * factor * B_mat @ C
#mats.A +  get_Q(n, nqs) * factor * mats.B @ mats.C

#############################################
# Core NQS Calculations                     #
#############################################

def _core(n, Ks, B, nqs, A, B_mat, C, factors,
          M = 1):
    """Compute bias and variance terms for dimension n.
    
    Args:
        n (int): Dimension index
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors
        bias (bool): Whether to compute bias term
        var (bool): Whether to compute variance term

    Returns:
        tuple: (bias, var) where bias is the bias term and var is the variance term
    """
    CC = jnp.kron(C, C)

    vterm = 0
    for (K, factor) in reversed(list(zip(Ks, factors))):
        T = get_T(n, nqs, A, B_mat, C, factor)
        TT = jnp.kron(T, T)
        Sk, TTk = superpower(TT, K)

        BB = jnp.kron(factor * B_mat, factor * B_mat)

        vterm = vterm + CC @ Sk @ BB
        CC = CC @ TTk

    v = get_V(n, B, nqs)
    q = get_Q(n, nqs)
    vterm = jnp.squeeze(0.5 * v * q * vterm)

    n_clipped = jnp.clip(n, M, None)

    ma = nqs[2]
    a = nqs[0]

    bterm = jnp.squeeze(0.5 * ma * n_clipped ** (-a) * jnp.sum(CC))
    #jnp.squeeze(0.5 * nqs.ma * n_clipped ** (-nqs.a) * jnp.sum(CC))

    return bterm, vterm


def _bv(n, Ks, B, nqs, A, B_mat, C, factors):
    """Compute risk contribution from dimension n.
    
    Args:
        n (int): Dimension index
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors
    
    Returns:
        float: Risk contribution from dimension n
    """
  # print("Compiling, time = ", time.time())
    bias, var = _core(n, Ks, B, nqs, A, B_mat, C, factors)
    return bias + var

def _grad_bv(n, Ks, B, nqs, A, B_mat, C, factors):
    """Compute the gradient of bias and variance with respect to all parameters.
    
    This function calculates the gradient of the bias+variance contribution from dimension n
    with respect to all six NQS parameters (a, b, ma, mb, c, sigma).
    
    Args:
        n (int): Dimension index
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors
    
    Returns:
        NQS: A namedtuple with the same structure as NQS, containing the gradients
             with respect to each parameter
    """
    # Define a function that computes b+v for a given NQS
    def bv_func(nqs_flat):
        return _bv(n, Ks, B, nqs_flat, A, B_mat, C, factors)
    
    # Flatten NQS parameters to a vector
    nqs_flat = nqs
    
    # Compute gradient using JAX's vector-valued gradient function
    grad_flat = jax.grad(bv_func)(nqs_flat)
    
    # Unflatten the gradient back to NQS structure
    return grad_flat

def _tail(N, nqs):
    """Compute approximation term and the irreducible error.
    
    Args:
        N (int): Problem dimension
        nqs (NQS): System parameters
    
    Returns:
        float: 0.5 * ma * zeta(a, N+1) + c
    """
    ma = nqs[2]
    a = nqs[0]
    c = nqs[4]

    return 0.5 * ma * jnp.squeeze(jax.scipy.special.zeta(a, N+1)) + c

def _tail_array(N, nqs_array):
    """Compute approximation term and the irreducible error.
    
    Args:
        N (int): Problem dimension
        nqs (NQS): System parameters
    
    Returns:
        float: 0.5 * ma * zeta(a, N+1) + c
    """
    ma = nqs_array[2]
    a = nqs_array[0]
    c = nqs_array[4]

    return 0.5 * ma * jnp.squeeze(jax.scipy.special.zeta(a, N+1)) + c

def _reduce(f):
    """Compute reduced bias plus variance.
    
    Args:
        N (int): Problem dimension
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors
    
    Returns:
        float: Final risk value
    """

    def reducedf(N, Ks, B, nqs, A, B_mat, C, factors):

        def bodyf(val):
            s, n = val
            return (s + f(n, Ks, B, nqs, A, B_mat, C, factors), n-1)

        def condf(val):
            s, n = val
            return n > 0

        risk, _ = jax.lax.while_loop(condf, bodyf, (0, N))
        return risk

    return reducedf


def _reduce_vec(f):
    """Compute reduced bias plus variance.
    
    Args:
        N (int): Problem dimension
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors
    
    Returns:
        float: Final risk value
    """

    def reducedf(N, Ks, B, nqs, A, B_mat, C, factors):

        def bodyf(val):
            s, n = val
            return (s + f(n, Ks, B, nqs, A, B_mat, C, factors), n-1)

        def condf(val):
            s, n = val
            return n > 0

        risk, _ = jax.lax.while_loop(condf, bodyf, (jnp.zeros(6), N))
        return risk

    return reducedf

_reducedbv = _reduce(_bv)

#############################################
# Derivatives                               #
#############################################

#_dbv_dn = jax.jacfwd(_bv)
#_dgrada_bv_dn = jax.jacfwd(_grada_bv)
#_dgradb_bv_dn = jax.jacfwd(_gradb_bv)
#_dgradma_bv_dn = jax.jacfwd(_gradma_bv)
#_dgradmb_bv_dn = jax.jacfwd(_gradmb_bv)
#_dgradsigma_bv_dn = jax.jacfwd(_gradsigma_bv)

#_ddbv_dndn = jax.jacfwd(_dbv_dn)
#_ddgrada_bv_dndn = jax.jacfwd(_dgrada_bv_dn)
#_ddgradb_bv_dndn = jax.jacfwd(_dgradb_bv_dn)
#_ddgradma_bv_dndn = jax.jacfwd(_dgradma_bv_dn)
#_ddgradmb_bv_dndn = jax.jacfwd(_dgradmb_bv_dn)
#_ddgradsigma_bv_dndn = jax.jacfwd(_dgradsigma_bv_dn)


_reducedgrad_bv = _reduce_vec(_grad_bv)

#def to_array(nqs):
#    """Convert NQS namedtuple to a flat array."""
 #   return jnp.array([nqs.a, nqs.b, nqs.ma, nqs.mb, nqs.c, nqs.sigma])

_gradnqs_tail_array = lambda N, nqs: jax.grad(_tail_array, 1)(N, nqs)



################################
# Get Training Loss            #
################################




################################
# Fitting                      #
    ############################

import os
import json
import numpy as np
from datetime import datetime
import pandas as pd



def _fit_multiple(data_tuples, lossarr, nqs0_list, return_traj=False, save_trajectories=True,
                output_dir="nqs_trajectories", compute_loss_only=False,
                steps=10, gtol=1e-8):
    
    """JAX-compatible version of the fit function with multiple initializations."""
    
    # Set default values
    num = len(data_tuples)
    n_inits = len(nqs0_list)
    

    # Helper function to process a single item - Ks is static
    @partial(jax.jit, static_argnums=(3,))
    def process_single_item(x, data_item, loss_val, Ks_static):
        """Process a single data point with static Ks."""
        N, _, B, A, B_mat, C, factors = data_item
        nqs = _to_nqs(x)
        
        # Use _fastrisk with static Ks
        risk_val = _fastrisk(N, Ks_static, B, nqs, A, B_mat, C, factors)
        
        log_loss = jnp.log(loss_val)
        log_risk = jnp.log(risk_val)

        huber_delta=1e-3
        log_difference = jnp.abs(log_loss - log_risk)

        # if the log_diff >= delta, huber_cond = 1 else 0
        huber_cond = jnp.maximum(0, (log_difference - huber_delta)/jnp.abs(log_difference - huber_delta))
        raw_loss = 0.5 * (log_loss - log_risk)**2 
        add_if_huber_cond = - raw_loss + huber_delta * (log_difference - 0.5 * huber_delta)
        huber_loss = raw_loss + huber_cond * add_if_huber_cond
        
        
        return huber_loss
    #0.5 * (log_loss - log_risk)**2
    
    # Also create a gradient version that preserves Ks as static
    @partial(jax.jit, static_argnums=(3,))
    def grad_single_item(x, data_item, loss_val, Ks_static):
        """Compute gradient for a single data point with static Ks."""
        N, _, B, A, B_mat, C, factors = data_item
        nqs = _to_nqs(x)
        
        # Calculate risk value
        risk_val = _fastrisk(N, Ks_static, B, nqs, A, B_mat, C, factors)
        
        # Calculate risk gradient - explicitly use Ks_static
        risk_grad = _fastgradrisk(N, Ks_static, B, nqs, A, B_mat, C, factors)
        
        # Chain rule for loss gradient
        log_factor = -(jnp.log(loss_val) - jnp.log(risk_val)) / risk_val

        

        log_loss = loss_val
        log_risk = risk_val
        log_difference = jnp.abs(log_loss - log_risk)

        huber_delta=1e-3
        # if the log_diff >= delta, huber_cond = 1 else 0
        huber_cond = jnp.maximum(0, (log_difference - huber_delta)/jnp.abs(log_difference - huber_delta))
        #raw_loss = 0.5 * (log_loss - log_risk)**2 
        #add_if_huber_cond = - raw_loss + huber_delta * (log_difference - 0.5 * huber_delta)
        #huber_loss = raw_loss + huber_cond * add_if_huber_cond
        
        huber_factor = - log_factor + huber_delta / risk_val * jnp.abs(log_risk - log_loss)/(log_risk - log_loss)
        
        mult_factor = log_factor + huber_cond * huber_factor
        # Convert NQS gradient to parameter gradient
        grad_x = jac_y_to_jac_ynorm(jac_nqs_to_jac_y(risk_grad, nqs), nqs_to_y(nqs))
        
        # Apply chain rule
        return mult_factor * grad_x
        #log_factor * grad_x
    
    # Pre-process data to group by unique Ks values
    # First, find all unique Ks values
    Ks_to_indices = {}
    # add loss_arr to data_tuples
    for i, data_item in enumerate(data_tuples):
        _, Ks, _, _, _, _, _ = data_item
        # Convert Ks to a hashable form if it's not already
        Ks_hashable = Ks if isinstance(Ks, tuple) else tuple(Ks)
        if Ks_hashable not in Ks_to_indices:
            Ks_to_indices[Ks_hashable] = []
        Ks_to_indices[Ks_hashable].append(i)
    
    # Create specialized loss and gradient functions for each Ks group
    Ks_function_pairs = []
    for Ks_hashable, indices in Ks_to_indices.items():
        # Get the actual Ks value (may be different from the hashable version)
        Ks_actual = data_tuples[indices[0]][1]
        
        # Filter data for this Ks value
        filtered_data = [data_tuples[i] for i in indices]
        #print("filtered_data = ", filtered_data)
        filtered_loss = jnp.array([lossarr[i] for i in indices])
        #print("filtered_loss = ", jnp.log(filtered_loss))
        # add loss to the data_tuples
        #filtered_data = [(lossarr[i],) + data_tuples[i] for i in indices]

        group_size = len(indices)
        
        # Create a specialized loss function for this Ks group
        @partial(jax.jit, static_argnums=(1,))
        def group_loss_fn(x, Ks_static=Ks_actual, filtered_data=filtered_data, filtered_loss=filtered_loss):
            # Process each item individually and sum
            total = 0.0
            for i in range(len(filtered_data)):
                loss = process_single_item(x, filtered_data[i], filtered_loss[i], Ks_static)
                total += loss
            return total
        
        # Create a specialized gradient function for this Ks group
        @partial(jax.jit, static_argnums=(1,))
        def group_grad_fn(x, Ks_static=Ks_actual, filtered_data=filtered_data, filtered_loss=filtered_loss):
            # Process each item individually and sum gradients
            grad_sum = jnp.zeros_like(x)
            for i in range(len(filtered_data)):
                grad = grad_single_item(x, filtered_data[i], filtered_loss[i], Ks_static)
                grad_sum += grad
            return grad_sum
        
        # Store both functions and the group size
        Ks_function_pairs.append((group_loss_fn, group_grad_fn, group_size))
    
    # Main loss function that combines results from all Ks groups
    @jax.jit
    def calc_total_loss(x):
        # Accumulate loss from each group
        total = 0.0
        for loss_fn, _, _ in Ks_function_pairs:
            total += loss_fn(x)
        return total / num




    # Vectorized loss function to process multiple initializations
    @jax.jit
    def batch_loss_fn(x_list):
        # Process each initialization in parallel
        losses = jax.vmap(calc_total_loss)(x_list)
        return losses

    if compute_loss_only:
        # Compute loss for all initializations
        losses = batch_loss_fn(jnp.stack([_to_x(nqs) for nqs in nqs0_list]))
        return losses

    # Custom gradient function that combines group-specific gradients
    @jax.jit
    def calc_grad(x):
        # Accumulate gradients from each group
        grad_total = jnp.zeros_like(x)
        for _, grad_fn, _ in Ks_function_pairs:
            grad_total += grad_fn(x)
        return grad_total / num
    
    # Define a step function for a single initialization
    
    @jax.jit
    def step_fn(carry, step_idx):
        nqs_params, m, v, x_0, active_flag = carry
        
        # Adam hyperparameters
        learning_rate = 0.001  # Base learning rate
        beta1 = 0.9  # Exponential decay rate for first moment
        beta2 = 0.999  # Exponential decay rate for second moment
        epsilon = 1e-8  # Small constant for numerical stability
        
        # Get parameters and compute gradient
        x = _to_x(nqs_params)
        grad = calc_grad(x)  # Use our custom gradient function
        
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
        
        # Convert back to NQS
        nqs_params_new = _to_nqs(x_new)
        
        # Check convergence conditions
        dist_from_init = jnp.linalg.norm(x_new - x_0)
        step_size = jnp.linalg.norm(x_new - x)
        grad_norm = jnp.linalg.norm(grad)
        
        # Create convergence flags
        converged_step = False  # step_size < gtol * 0.01
        converged_grad = grad_norm < gtol
        too_far = False  # dist_from_init > max_shift
        
        # Update active flag - we continue if still active and not converged
        new_active_flag = active_flag & ~(converged_step | converged_grad | too_far)
        
        # Collect data for trajectory
        traj_info = (nqs_params_new, x_new, calc_total_loss(x_new), 
                    grad_norm, step_size, dist_from_init, new_active_flag)
        
        # Return updated state
        return (nqs_params_new, m_new, v_new, x_0, new_active_flag), traj_info

    @jax.jit
    # Vectorize the step function over multiple initializations
    def batch_step_fn(batch_state, step_idx):
        batch_results = jax.vmap(lambda state: step_fn(state, step_idx))(batch_state)
        return batch_results

    # Initialize optimization state for each starting point
    init_x = jnp.stack([_to_x(nqs) for nqs in nqs0_list])
    init_m = jnp.zeros_like(init_x)  # First moment estimate (momentum)
    init_v = jnp.zeros_like(init_x)  # Second moment estimate (velocity)
    init_nqs_params = jnp.stack(nqs0_list)
    init_active_flags = jnp.ones(n_inits, dtype=bool)  # All trajectories start active

    init_state = (init_nqs_params, init_m, init_v, init_x, init_active_flags)

    # Use scan instead of fori_loop for better differentiability
    @jax.jit
    def scan_step(state_and_traj, i):
        state, traj_list = state_and_traj
        
        # Run one step of optimization
        new_state, new_traj_info = batch_step_fn(state, i)
        
        # For scan, we need to return the same structure as the input
        return (new_state, traj_list), new_traj_info
    

    # Run the optimization with scan and collect trajectories
    (final_state, _), traj_infos = jax.lax.scan(
        scan_step,
        (init_state, []),  # Initial state
        jnp.arange(steps)   # Iteration indices
    )
    
    # Convert trajectory info to list
    trajectories = list(traj_infos)
    
    # Get final results
    final_nqs_params, _, _, _, _ = final_state
    final_losses = jax.vmap(lambda nqs: calc_total_loss( _to_x(nqs)))(final_nqs_params)
    #best_idx = jnp.argmin(final_losses)
    # get best idx but ignore nans
    best_idx = jnp.nanargmin(final_losses)
    best_nqs = final_nqs_params[best_idx]
    

    # Convert trajectory info to list
    # Convert trajectory info to list
    trajectories = list(traj_infos)

    # Inspect trajectories
    #inspect_trajectory_structure(trajectories)


    # Save trajectories if requested
    if save_trajectories:
        traj_to_return = save_optimization_trajectories(trajectories, nqs0_list, output_dir)
    
    # Return results based on return_traj flag
    if return_traj:
        return best_nqs, traj_to_return
    else:
        return best_nqs
    
 


##############################################
# Numerical Integration with JAX
##############################################

def jax_quad(f, a, b, epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
    """
    JAX implementation of numerical integration using fixed-point Gaussian quadrature
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        epsabs: Absolute error tolerance (not used in this implementation)
        epsrel: Relative error tolerance (not used in this implementation)
        limit: Number of evaluation points (determines precision)
        
    Returns:
        (result, error_estimate)
    """
    # Limit is used to determine the degree of quadrature 
    # We'll use a simple fixed-point Gauss-Legendre rule
    
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
    y = jax.vmap(f)(mapped_x) 
    
    # Compute integral
    result = radius * jnp.sum(w * y)
    
    # Simple error estimate - in a real adaptive rule, this would be
    # based on the difference between two rules of different order
    #error_est = jnp.abs(result) * 1e-6
    
    # get the device information for result
    #device = jax.devices()[0]
    # turn result into a 1D array
    
    return result #, error_est


def jax_quad_vec(f, a, b, epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
    """
    JAX implementation of numerical integration using fixed-point Gaussian quadrature
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        epsabs: Absolute error tolerance (not used in this implementation)
        epsrel: Relative error tolerance (not used in this implementation)
        limit: Number of evaluation points (determines precision)
        
    Returns:
        (result, error_estimate)
    """
    # Limit is used to determine the degree of quadrature 
    # We'll use a simple fixed-point Gauss-Legendre rule
    
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

    
    # Compute integral
    result = radius * jnp.sum(w[:, jnp.newaxis] * y_values, axis=0)
    
    # Simple error estimate - in a real adaptive rule, this would be
    # based on the difference between two rules of different order
    #error_est = jnp.abs(result) * 1e-6
    
    # get the device information for result
    #device = jax.devices()[0]
    return result #, error_est



def _em(N, g, 
        headsum, tailsum, p=1, epsrel=1e-2, vec = False):
    """Euler-Maclaurin approximation for numerical integration in NQS calculations.
    
    Simplified implementation to avoid tracer issues.
    """
    if not p in [1]:
        raise ValueError(f"Euler-Maclaurin order p={p} is not supported.")

    def f(x):
        n = jnp.exp(x)
        dn = jnp.exp(x)
        return g(n) * dn

    # Fixed cutoff at 10% of N to avoid any conditional logic
    # This avoids tracer issues with dynamic choices
    M = jnp.minimum(jnp.maximum(1, jnp.array((N * 0.05), int)), 100)
    
    L = jnp.log(M)
    U = jnp.log(N)
    
    if vec:
        integral = jax_quad_vec(f, L, U, epsrel=epsrel)
    else:
        integral = jax_quad(f, L, U, epsrel=epsrel)
    
    # Compute risk
    risk = headsum(M)
    risk += integral
    risk += (g(N) - g(M)) / 2


    return risk + tailsum(N)




def _fastrisk(N, Ks, B, nqs, A, B_mat, C, factors):
    """Compute risk using fast quadrature approximation.

    Args:
        N (int): Problem dimension
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors

    Returns:
        float: Approximated risk value
    """
    return _em(N,
        lambda n: jax.jit(_bv, static_argnames=("Ks",))(n, Ks, B, nqs, A, B_mat, C, factors),
        lambda M: jax.lax.cond(
            M > 0,
            lambda _: jax.jit(_reducedbv, static_argnames=("Ks",))(M, Ks, B, nqs, A, B_mat, C, factors),
            lambda _: jnp.array(0.),
            operand=None
            ),
        lambda N: jax.jit(_tail)(N, nqs)
    )


def _fastgradrisk(N, Ks, B, nqs, A, B_mat, C, factors):
    """Compute the gradient of risk with respect to NQS parameters using fast quadrature.

    Args:
        N (int): Problem dimension
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors

    Returns:
        NQS: A namedtuple with the same structure as NQS, where each field contains
            the gradient of risk with respect to that parameter
    """
    dall = _em(N,
        lambda n: jax.jit(_grad_bv, static_argnames=("Ks",))(n, Ks, B, nqs, A, B_mat, C, factors),
        lambda M: jax.lax.cond(
            M > 0,
            lambda _: jax.jit(_reducedbv, static_argnames=("Ks",))(M, Ks, B, nqs, A, B_mat, C, factors),
            lambda _: jnp.array(0.),
            operand=None
            ),
        lambda N: jax.jit(_gradnqs_tail_array)(N, nqs),
        vec=True
    )


    
    return dall







# ----------------#
# Utilities
# ----------------#
@partial(jax.jit, static_argnames=('n',))
def superpower(A, n):
    """Compute the matrix power series sum S(A,n) and power A^n efficiently using binary decomposition.
        
    This function computes two values:
    1. S(A,n) = I + A + A^2 + ... + A^(n-1)  (the power series sum)
    2. A^n (the nth power of matrix A)
    
    Args:
        A (jax.numpy.ndarray): Input matrix or batch of matrices. The last two dimensions
            represent the matrix dimensions.
        n (int): Non-negative integer power to compute.
    
    Returns:
        tuple: A pair (S, P) where:
            - S is the power series sum S(A,n) = I + A + A^2 + ... + A^(n-1)
            - P is the matrix power A^n
    
    Example:
        >>> import jax.numpy as jnp
        >>> A = jnp.array([[1., 2.], [3., 4.]])
        >>> S, P = superpower(A, 3)
        >>> # S will be I + A + A^2
        >>> # P will be A^3
    """
    if n == 0:
        Z = jnp.zeros_like(A)
        I = jnp.broadcast_to(jnp.eye(A.shape[-2], dtype=A.dtype), A.shape)
        return (Z, I)
    else:
        k, bit = divmod(n, 2)
        Sk, Ak = superpower(A, k)
        A2k = Ak @ Ak
        if bit == 0:
            return (Sk + Sk @ Ak, A2k)
        else:
            return (Sk + Sk @ Ak + A2k, A2k @ A)

def _process_milestones(K, milestones):
    """Process milestones for learning rate schedulers.

    Ensures the milestone list starts with 0 and ends with K, and includes only
    milestone points less than K.

    Args:
        K (int): Total number of iterations
        milestones (List[int]): List of milestone iteration indices

    Returns:
        List[int]: Processed milestone list
    """

    new_milestones = [0]
    i = 0
    while i < len(milestones) and milestones[i] < K:
        new_milestones.append(milestones[i])
        i += 1
    new_milestones.append(K)
    return new_milestones


def make_schedule(K, scheduler): #folds):
    """Creates a learning rate schedule for K iterations.
    
    Args:
        K (int): Total number of iterations
        folds (FOLDS): Contains the scheduler specification
    
    Returns:
        tuple: (steps, factors) where:
            - steps is sequence of iteration counts for each piece
            - factors is sequence of learning rate factors for each piece
    """
    if K == 0:
        return (), ()

    match scheduler:
        case None:
            Ks = [K]
            factors = [1.]

        case ConstantLR(factor=factor, total_iters=total_iters):
            Ks = [min(total_iters, K)]
            factors = [factor]
            if total_iters <= K:
                Ks.append(K - total_iters)
                factors.append(1.0)

        case StepLR(step_size=span_size, gamma=gamma):
            n_spans = K // span_size
            Ks = [span_size for _ in range(n_spans)]
            factors = [gamma ** i for i in range(n_spans)]
            # leftover
            if K % span_size > 0:
                Ks.append(K % span_size)
                factors.append(gamma ** n_spans)

        case MultiStepLR(milestones=milestones, gamma=gamma):
            Ks = []
            factors = []
            milestones = _process_milestones(K, milestones)
            i = 1
            while i < len(milestones):
                Ki = milestones[i] - milestones[i-1]
                Ks.append(Ki)
                factors.append(gamma ** (i-1))
                i += 1

        case SequentialLR(milestones=milestones, schedulers=subschedulers):
            Ks = []
            factors = []
            milestones = _process_milestones(K, milestones)
            i = 1
            while i < len(milestones):
                Ki = milestones[i] - milestones[i-1]
                scheduler_idx = min(i, len(subschedulers))-1
                subKs, subfactors = make_schedule(Ki, subschedulers[scheduler_idx])
                Ks.extend(subKs)
                factors.extend(subfactors)
                i += 1

    return tuple(Ks), tuple(factors)

    
def get_reparam_funcs():
    # Reparametrization functions
    def nqs_to_y(nqs):
            # reparametrize nqs
            a = nqs.a
            b = nqs.b
            ma = nqs.ma
            mb = nqs.mb
            c = nqs.c
            sigma = nqs.sigma

            ap = a/b - 1/b
            bp = b
            ra = 1/ma**(1/(a-1))
            rb = mb/(ma**(1/(a/b-1/b)))

            return jnp.array([ap, bp, ra, rb, c, sigma])
        
    def y_to_ynorm(y, norm_const = 1e5):
            # reparametrize nqs
            ap = y[0]
            bp = y[1]
            ra = y[2]
            rb = y[3]
            c = y[4]
            sigma = y[5]

            ra = jnp.log(ra)/jnp.log(norm_const)
            rb = jnp.log(rb)/jnp.log(norm_const)
            c = jnp.log(c)/jnp.log(norm_const)
            sigma = jnp.log(sigma)/jnp.log(norm_const)

            return jnp.array([ap, bp, ra, rb, c, sigma])

    def ynorm_to_y(y, norm_const = 1e5):
            # reparametrize nqs
            ap = y[0]
            bp = y[1]
            ra = y[2]
            rb = y[3]
            c = y[4]
            sigma = y[5]

            ra = norm_const ** ra
            rb = norm_const ** rb
            c = norm_const ** c 
            sigma = norm_const ** sigma 

            return jnp.array([ap, bp, ra, rb, c, sigma])

    def y_to_nqs(y):
            # reparametrize nqs
            ap = y[0]
            bp = y[1]
            ra = y[2]
            rb = y[3]
            c = y[4]
            sigma = y[5]

            a = ap*bp + 1
            b = bp
            ma = (1/ra) ** (ap*bp)
            mb = rb/(ra**bp)
            
            return NQS(a=a, b=b, ma=ma, mb=mb, c=c, sigma=sigma)

    def jac_nqs_to_jac_y(jac_nqs, nqs):
            # reparametrize nqs
            da = jac_nqs[0]
            db = jac_nqs[1]
            dma = jac_nqs[2]
            dmb = jac_nqs[3]
            dc = jac_nqs[4]
            dsigma = jac_nqs[5]

            a = nqs.a
            b = nqs.b
            ma = nqs.ma
            mb = nqs.mb
            c = nqs.c
            sigma = nqs.sigma

            nqsp = nqs_to_y(nqs)
            ap = nqsp[0]
            bp = nqsp[1]
            ra = nqsp[2]
            rb = nqsp[3]

            dap = da * bp + dma * jnp.log(1/ra)*(1/ra)**(ap*bp) * bp
            dbp = da*ap + db + dma*jnp.log(1/ra)*(1/ra)**(ap*bp)* ap + dmb * rb * jnp.log(1/ra)*(1/(ra**bp))
            dra = dma*(-ap*bp)*ra**(-ap*bp-1) + dmb*rb*(-bp)*ra**(-bp-1)
            drb = dmb/(ra**bp)

            return jnp.array([dap, dbp, dra, drb, dc, dsigma])

    def jac_y_to_jac_ynorm(jac_y, y, norm_const = 1e5):
            dap = jac_y[0]
            dbp = jac_y[1]
            dra = jac_y[2]
            drb = jac_y[3]
            dc = jac_y[4]
            dsigma = jac_y[5]

            ap = y[0]
            bp = y[1]
            ra = y[2]
            rb = y[3]
            c = y[4]
            sigma = y[5]

            dra = dra * ra * jnp.log(norm_const)
            drb = drb * rb * jnp.log(norm_const)
            dc = dc * c * jnp.log(norm_const)
            dsigma = dsigma * sigma * jnp.log(norm_const)

            return jnp.array([dap, dbp, dra, drb, dc, dsigma])

    def jac_ynorm_to_jac_y(jac_ynorm, ynorm, norm_const = 1e5):
            dap = jac_ynorm[0]
            dbp = jac_ynorm[1]
            dra = jac_ynorm[2]
            drb = jac_ynorm[3]
            dc = jac_ynorm[4]
            dsigma = jac_ynorm[5]

            y = ynorm_to_y(ynorm)
            ap = y[0]
            bp = y[1]
            ra = y[2]
            rb = y[3]
            c = y[4]
            sigma = y[5]

            dra = dra / ra / jnp.log(norm_const)
            drb = drb / rb / jnp.log(norm_const)
            dc = dc / c / jnp.log(norm_const)
            dsigma = dsigma / sigma / jnp.log(norm_const)

            return jnp.array([dap, dbp, dra, drb, dc, dsigma])
        
    def jac_y_to_jac_nqs(jac_y, y):
            dap = jac_y[0]
            dbp = jac_y[1]
            dra = jac_y[2]
            drb = jac_y[3]
            dc = jac_y[4]
            dsigma = jac_y[5]

            nqss = y_to_nqs(y)
            a = nqss.a
            b = nqss.b
            ma = nqss.ma
            mb = nqss.mb
            c = nqss.c
            sigma = nqss.sigma

            da = dap/b + dra*jnp.log(1/ma)*(1/ma)**(1/(a-1))*(-1/(a-1)**2) + drb*mb *jnp.log(ma)*ma**(-1/(a/b -1/b))*(1/(a/b-1/b)**2)*1/b
            db = dap*(a-1)*(-1/b**2) + dbp + drb*mb*jnp.log(ma)*ma**(-1/(a/b-1/b))*(1/(a/b-1/b)**2)*(-1/b**2) *(a-1)
            dma = dra*(-1/(a-1))*ma**(-1/(a-1)-1) + drb*mb*(-1/(a/b-1/b))*ma**(-1/(a/b-1/b)-1)
            dmb = drb*(1/ma**(1/(a/b-1/b)))

            return NQS(a=da, b=db, ma=dma, mb=dmb, c=dc, sigma=dsigma)

    def nqs_to_params_dict(nqs):
        dictt = {"a": nqs.a, "b": nqs.b, "ma": nqs.ma, 
                "mb": nqs.mb, "c": nqs.c, "sigma": nqs.sigma}
        y = nqs_to_y(nqs)
        dictt["ap"] = y[0]
        dictt["bp"] = y[1]
        dictt["ra"] = y[2]
        dictt["rb"] = y[3]
        return dictt

    return nqs_to_y, y_to_ynorm, ynorm_to_y, y_to_nqs, jac_nqs_to_jac_y, jac_y_to_jac_ynorm, jac_ynorm_to_jac_y, jac_y_to_jac_nqs, nqs_to_params_dict


#############################################
# Basic Objects                             #
#############################################

NQS = namedtuple("NQS", ["a", "b", "ma", "mb", "c", "sigma"])
"""Parameters defining a Noisy Quadratic System.

Attributes:
    a (float): Decay parameter for the curvature (must be > 1)
    b (float): Decay parameter for the minimum (must be > 0)
    ma (float): Scale parameter for the curvature (must be > 0)
    mb (float): Scale parameter for the minimum (must be > 0)
    c (float): Constant offset term (must be >= 0)
    sigma (float): Standard deviation of the noise (must be > 0)
"""

mats = namedtuple("mats", ["A", "B", "C"])
"""Matrices defining a linear dynamical system.

Attributes:
    A (ndarray): State transition matrix
    B (ndarray): Input matrix
    C (ndarray): Output matrix
"""

ConstantLR = namedtuple("ConstantLR", ["factor", "total_iters"])
"""Constant learning rate scheduler.

Attributes:
    factor (float): Learning rate multiplier
    total_iters (int): Number of iterations to maintain this rate
"""

StepLR = namedtuple("StepLR", ["step_size", "gamma"])
"""Step learning rate scheduler that decays by gamma every step_size iterations.

Attributes:
    step_size (int): Number of iterations between learning rate updates
    gamma (float): Multiplicative factor for learning rate decay
"""

MultiStepLR = namedtuple("MultiStepLR", [ "milestones", "gamma"])
"""Multi-step learning rate scheduler that decays at specified milestones.

Attributes:
    milestones (List[int]): Iterations at which to decay learning rate
    gamma (float): Multiplicative factor for learning rate decay
"""

SequentialLR = namedtuple("SequentialLR", ["milestones", "schedulers"])
"""Sequential learning rate scheduler that switches between different schedulers.

Attributes:
    milestones (List[int]): Iterations at which to switch schedulers
    schedulers (List): List of scheduler objects to use sequentially
"""

FOLDS = namedtuple("FOLDS", ["mats", "scheduler"])
"""First-Order Linear Dynamical System specification.

Attributes:
    mats (mats): Matrices A, B, C defining the linear system
    scheduler: Learning rate scheduler object
"""

def SGD(lr, momentum, nesterov, scheduler):
    """Creates a FOLDS representation of SGD with optional momentum.
    
    Args:
        lr (float): Learning rate
        momentum (float): Momentum coefficient (0 for vanilla SGD)
        nesterov (bool): Whether to use Nesterov momentum
        scheduler (Union[None, ConstantLR, StepLR, MultiStepLR, SequentialLR]): Learning rate scheduler
    
    Returns:
        FOLDS: Named tuple containing matrices A, B, C defining the linear dynamical system
               and the learning rate scheduler
    """
    A = jnp.array([[1.0 + momentum, -momentum],
                   [1.0, 0.0]])
    B = jnp.array([[-lr],
                   [0.0]])
    if nesterov:
        C = jnp.array([[1.0 + momentum, -momentum]])
    else:
        C = jnp.array([[1.0, 0.0]])
    return FOLDS(mats(A, B, C), scheduler)

# ---------------------------------------- #
# Define functions to be used with Objects
# ---------------------------------------- #

# risk: this is wrapper over _fastrisk, but instead of nqs_array, A, B_mat, C and factors, it 
# takes nqs, and folds objects
def fastrisk(N, K, B, nqs, folds):
    """Compute risk using fast quadrature approximation.

    Args:
        N (int): Problem dimension
        K (int): Number of iterations
        B (int): Mini-batch size
        nqs (NQS): System parameters
        folds (FOLDS): Contains the scheduler specification

    Returns:
        float: Approximated risk value
    """
    A = folds.mats.A
    B_mat = folds.mats.B
    C = folds.mats.C
    Ks, factors = make_schedule(K, folds.scheduler)

    nqs_array = jnp.array([nqs.a, nqs.b, nqs.ma, nqs.mb, nqs.c, nqs.sigma])

    return _fastrisk(N, Ks, B, nqs_array, A, B_mat, C, factors)

def gradrisk(N, K, B, nqs, folds, kind = "fast"):

    if kind != "fast":
        raise ValueError(f"Unknown risk calculation type: {kind}")
    A = folds.mats.A
    B_mat = folds.mats.B
    C = folds.mats.C
    Ks, factors = make_schedule(K, folds.scheduler)
    nqs_array = jnp.array([nqs.a, nqs.b, nqs.ma, nqs.mb, nqs.c, nqs.sigma])

    out_array = _fastgradrisk(N, Ks, B, nqs_array, A, B_mat, C, factors)
    out_nqs = NQS(a=out_array[0], b=out_array[1], ma=out_array[2], mb=out_array[3], c=out_array[4], sigma=out_array[5])
    return out_nqs

# wrapper over fit multiple, but takes in NBKfoldsarr of the objects format,
# and reformats it for _fit_multiple
def fit_multiple(NKBfoldsarrs, lossarr, nqs0_list=None, return_traj=False, 
                steps=10, gtol=1e-8):
    
    NKBfoldsarrs_flat = []
    for NKBfolds in NKBfoldsarrs:
        N, K, B, folds = NKBfolds
        folds_mats = folds.mats
        A = folds_mats.A
        B_mat = folds_mats.B
        C = folds_mats.C
        scheduler = folds.scheduler
        Ks, factors = make_schedule(K, scheduler)
        flatted_NKBfolds = (N, Ks, B, A, B_mat, C, factors)
        NKBfoldsarrs_flat.append(flatted_NKBfolds)
    
    nqs0_list_flat = []
    for nqs0 in nqs0_list:
        nqs0_flat = jnp.array([nqs0.a, nqs0.b, nqs0.ma, nqs0.mb, nqs0.c, nqs0.sigma])
        nqs0_list_flat.append(nqs0_flat)

    outs =  _fit_multiple(NKBfoldsarrs_flat, lossarr, nqs0_list_flat,
                        return_traj=return_traj, steps=steps, gtol=gtol)
    
    if return_traj:
        fitted_nqs, trajectories = outs
        fitted_nqs_formatted = NQS(a=fitted_nqs[0], b=fitted_nqs[1], ma=fitted_nqs[2], mb=fitted_nqs[3], c=fitted_nqs[4], sigma=fitted_nqs[5])
        #trajectories_formatted = [NQS(a = nqs_array[0], b = nqs_array[1], ma = nqs_array[2], mb = nqs_array[3], c = nqs_array[4], sigma = nqs_array[5]) for nqs_array in trajectories]
        trajectories_formatted = []
        for traj_dict in trajectories:
            traj_dict_formatted = traj_dict.copy()
            traj = traj_dict_formatted["trajectory"]
            traj_formatted = [NQS(a = nqs_array[0], b = nqs_array[1], ma = nqs_array[2], mb = nqs_array[3], c = nqs_array[4], sigma = nqs_array[5]) 
                              for nqs_array in traj]
            # update the dictionary with the formatted trajectories
            traj_dict_formatted["trajectory"] = traj_formatted
            # append the formatted dictionary to the list
            trajectories_formatted.append(traj_dict_formatted)


        return fitted_nqs_formatted, trajectories_formatted
    else:
        fitted_nqs = outs
        fitted_nqs_formatted = NQS(a=fitted_nqs[0], b=fitted_nqs[1], ma=fitted_nqs[2], mb=fitted_nqs[3], c=fitted_nqs[4], sigma=fitted_nqs[5])
        return fitted_nqs_formatted
    

def compute_loss_multiple(NKBfoldsarrs, lossarr, nqs0_list):
    # convert nqs0_list to nqs0_list_flat
    nqs0_list_flat = [jnp.array([nqs0.a, nqs0.b, nqs0.ma, nqs0.mb, nqs0.c, nqs0.sigma]) 
                      for nqs0 in nqs0_list]

    # convert NKBfoldsarrs to NKBfoldsarrs_flat
    NKBfoldsarrs_flat = []
    for NKBfolds in NKBfoldsarrs:
        N, K, B, folds = NKBfolds
        folds_mats = folds.mats
        A = folds_mats.A
        B_mat = folds_mats.B
        C = folds_mats.C
        scheduler = folds.scheduler
        Ks, factors = make_schedule(K, scheduler)
        flatted_NKBfolds = (N, Ks, B, A, B_mat, C, factors)
        NKBfoldsarrs_flat.append(flatted_NKBfolds)
    # compute loss using _fit_multiple, compute_loss_only = True
    loss_at_nqs_list = _fit_multiple(NKBfoldsarrs_flat, lossarr, nqs0_list_flat,
                        compute_loss_only=True)
    
    return loss_at_nqs_list




                          

# risk a wrapper over fastrisk, but takes in an additional argument kind = "fast"
def risk(N, K, B, nqs, folds, kind="fast"):
    """Compute risk using fast quadrature approximation.

    Args:
        N (int): Problem dimension
        K (int): Number of iterations
        B (int): Mini-batch size
        nqs (NQS): System parameters
        folds (FOLDS): Contains the scheduler specification
        kind (str): Type of risk calculation ("fast" or "grad")

    Returns:
        float: Approximated risk value
    """
    if kind == "fast":
        return fastrisk(N, K, B, nqs, folds)
    else:
        raise ValueError(f"Unknown risk calculation type: {kind}")

def to_x(nqs):
    # convert nqs to nqs_array
    nqs_array = jnp.array([nqs.a, nqs.b, nqs.ma, nqs.mb, nqs.c, nqs.sigma])
    # convert nqs_array to x using _to_x
    x = _to_x(nqs_array)
    return x

def to_nqs(x):
    # convert x to nqs_array
    nqs_array = _to_nqs(x)
    # convert nqs_array to nqs using y_to_nqs
    nqs = NQS(a=nqs_array[0], b=nqs_array[1], ma=nqs_array[2], mb=nqs_array[3], c=nqs_array[4], sigma=nqs_array[5])
    return nqs

#########################
# Saving Trajectories   #
#########################

def save_optimization_trajectories(trajectories, nqs0_list, output_dir="nqs_trajectories"):
    """Save optimization trajectories to disk in a structured format.
    
    Args:
        trajectories: JAX array of trajectory information with transposed structure
        nqs0_list: List of initial NQS values that were optimized
        output_dir: Directory to save trajectory files
    
    Returns:
        str: Path to the created trajectory directory
    """
    # Create timestamp for unique directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Get dimensions
    n_inits = len(nqs0_list)
    n_steps = trajectories[0].shape[0] if len(trajectories) > 0 else 0
    
    # Save metadata about this optimization run
    metadata = {
        "timestamp": timestamp,
        "num_initializations": n_inits,
        "num_steps": n_steps,
        "initial_nqs": []
    }
    
    # Convert and save initial NQS parameters
    for nqs in nqs0_list:
        params_dict = {}
        for key, value in nqs_to_params_dict(nqs).items():
            params_dict[key] = float(value) if isinstance(value, (jnp.ndarray, np.ndarray)) else value
        metadata["initial_nqs"].append(params_dict)
    
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Based on inspection, our trajectories structure is:
    # [
    #   nqs_params for all inits across all steps,
    #   x_values for all inits across all steps,
    #   loss_values for all inits across all steps,
    #   grad_norms for all inits across all steps,
    #   step_sizes for all inits across all steps,
    #   distances for all inits across all steps,
    #   active_flags for all inits across all steps
    # ]
    
    if len(trajectories) >= 7:  # Make sure we have the expected number of elements
        nqs_params_all = jax.device_get(trajectories[0])
        x_values_all = jax.device_get(trajectories[1])
        loss_values_all = jax.device_get(trajectories[2])
        grad_norms_all = jax.device_get(trajectories[3])
        step_sizes_all = jax.device_get(trajectories[4])
        distances_all = jax.device_get(trajectories[5])
        active_flags_all = jax.device_get(trajectories[6])
        
        # For each step and initialization, save a file
        for init_idx in range(n_inits):
            #for step_idx in range(n_steps):
            #    # Create a filename based on step and initialization index
            #    filename = os.path.join(run_dir, f"init_{init_idx}_step_{step_idx:04d}.npz")
                
            #    # Extract data for this step and initialization
            #    init_data = {
            #        "nqs_params": nqs_params_all[step_idx][init_idx],
            #        "x_values": x_values_all[step_idx][init_idx],
            #        "loss_values": loss_values_all[step_idx][init_idx],
            #        "grad_norms": grad_norms_all[step_idx][init_idx],
            #        "step_sizes": step_sizes_all[step_idx][init_idx],
            #        "distances": distances_all[step_idx][init_idx],
            #        "active_flags": active_flags_all[step_idx][init_idx]
            #    }

                
                
            #    # Save to file
            #    np.savez(filename, **init_data)

            # for each init, create a CSV file with all steps for this initialization
            csv_filename = os.path.join(run_dir, f"init_{init_idx}_trajectory.csv")


            # Collect data for all steps
            steps_data = []
            for step_idx in range(n_steps):
                step_row = {
                    # Include step number
                    'step': step_idx,
                    
                    # Include the 6 NQS parameters
                    'a': float(nqs_params_all[step_idx][init_idx][0]),
                    'b': float(nqs_params_all[step_idx][init_idx][1]),
                    'ma': float(nqs_params_all[step_idx][init_idx][2]),
                    'mb': float(nqs_params_all[step_idx][init_idx][3]),
                    'c': float(nqs_params_all[step_idx][init_idx][4]),
                    'sigma': float(nqs_params_all[step_idx][init_idx][5]),
                    
                    # Include normalized x values (flattened)
                    'x_0': float(x_values_all[step_idx][init_idx][0]),
                    'x_1': float(x_values_all[step_idx][init_idx][1]),
                    'x_2': float(x_values_all[step_idx][init_idx][2]),
                    'x_3': float(x_values_all[step_idx][init_idx][3]),
                    'x_4': float(x_values_all[step_idx][init_idx][4]),
                    'x_5': float(x_values_all[step_idx][init_idx][5]),
                    
                    # Include metrics
                    'loss': float(loss_values_all[step_idx][init_idx]),
                    'grad_norm': float(grad_norms_all[step_idx][init_idx]),
                    'step_size': float(step_sizes_all[step_idx][init_idx]),
                    'distance': float(distances_all[step_idx][init_idx]),
                    'active': bool(active_flags_all[step_idx][init_idx])
                }
                steps_data.append(step_row)

            # Create a DataFrame and save as CSV
            df = pd.DataFrame(steps_data)
            df.to_csv(csv_filename, index=False)
            print(f"CSV trajectory saved to: {csv_filename}")


    # Save final summary
    if len(trajectories) >= 3:  # Need at least params and losses
        final_nqs_params = np.array(jax.device_get(trajectories[0]))[-1, :, :]  # Last step for all inits
        final_losses = np.array(jax.device_get(trajectories[2]))[-1, :]  # Last step for all inits
        #best_idx = int(np.argmin(final_losses))
        #best_idx = int(np.nanargmin(final_losses))
        best_idx = int(np.nanargmin(final_losses))
        
        # Create structured data for all final NQS parameters
        all_final_nqs = []
        for idx in range(n_inits):
            params = {}
            for j, param_name in enumerate(['a', 'b', 'ma', 'mb', 'c', 'sigma']):
                params[param_name] = float(final_nqs_params[idx][j])
            params["final_loss"] = float(final_losses[idx])
            params["is_best"] = (idx == best_idx)
            all_final_nqs.append(params)
        
        summary = {
            "best_initialization": best_idx,
            "final_losses": [float(x) for x in final_losses],
            "all_final_parameters": all_final_nqs
        }
        
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # find the initialization with the best loss
    # do not include nan's in the argmin
    # best_idx = int(np.argmin(final_losses))
    best_idx = int(np.nanargmin(final_losses))
    # get the trajectory for this initialization: a list of nqs arrays
    best_trajectory = []
    for step_idx in range(n_steps):
        best_trajectory.append(nqs_params_all[step_idx][best_idx])
    
    # do the same for all trajectories (incl. the best one)
    # sort the list according to the loss, the best one first
    
    # first, sort the indices according to the loss
    sorted_indices = np.argsort(final_losses)
    # then, create a list of trajectories
    all_trajectories = []
    for idx in sorted_indices:
        traj_at_idx = []
        for step_idx in range(n_steps):
            traj_at_idx.append(nqs_params_all[step_idx][idx])
        # get the loss for this trajectory
        traj_loss = final_losses[idx]
        all_trajectories.append({
            "trajectory": traj_at_idx,
            "loss": traj_loss,
            "is_best": (idx == best_idx),
            "initialization": nqs0_list[idx],
            "initialization_index": idx}
        )



    # Create a README file explaining the format
    readme_text = f"""
    # NQS Optimization Trajectories
    
    This directory contains trajectory data for an NQS optimization run.
    
    ## Files
    
    - `metadata.json`: Information about the optimization run
    - `summary.json`: Summary of final results
    - `init_X_step_YYYY.npz`: Trajectory data for initialization X at step YYYY
    
    ## NPZ File Structure
    
    Each `.npz` file contains the following arrays:
    - `nqs_params`: NQS parameters [a, b, ma, mb, c, sigma]
    - `x_values`: Normalized parameter values
    - `loss_values`: Loss value at this step
    - `grad_norms`: Gradient norm
    - `step_sizes`: Size of the step taken
    - `distances`: Distance from initial parameters
    - `active_flags`: Boolean indicating if initialization is still active
    
    ## Loading Data
    
    ```python
    import numpy as np
    data = np.load('init_0_step_0001.npz')
    nqs_params = data['nqs_params']
    loss = data['loss_values']
    ```
    
    Generated: {timestamp}
    """
    
    with open(os.path.join(run_dir, "README.md"), "w") as f:
        f.write(readme_text)
    
    print(f"Trajectories saved to: {run_dir}")
    return all_trajectories



if __name__ == "__main__":


    def check_device(result):
        # Get the device where the result is located
        device = result.devices()
        print(f"Result is on: {device}")

    # Use it with your function
    def f(x):
        return x**2
    a = 0.0
    b = 1.0
    result = jax_quad(f, a, b)
    is_on_gpu = check_device(result) 

    if True:

        K = int(1e5)
        # K = int(10)
        # N = int(405e9)
        N = int(1e6)
        B=10
        nqs = NQS(a=2., b=2., ma=1.0, mb=1.0, c=0.01, sigma=0.01)
        scheduler = None
        folds = SGD(lr=0.1, momentum=0.0, nesterov=False, 
                    scheduler=scheduler)
       # raise ValueError(folds.mats)
        #Ks, factors = make_schedule(K, folds.scheduler)

        # test fastrisk
        # time 
        start = time.time()
        riskk = fastrisk(N, K, B, nqs, folds)
        print("riskk", riskk)
        end = time.time()
        print("time", end-start)

        start = time.time()
        gradd = gradrisk(N, K, B, nqs, folds)
        print("gradd", gradd)
        end = time.time()
        print("time", end-start)

        # switch out the NQS
        nqs = NQS(a=1.5, b=1., ma=1.0, mb=1.0, c=0.01, sigma=0.9)
        # compute the risk
        start = time.time()
        riskk = fastrisk(N, K, B, nqs, folds)
        print("riskk", riskk)
        end = time.time()
        print("time", end-start)

        start = time.time()
        gradd = gradrisk(N, K, B, nqs, folds)
        print("gradd", gradd)
        end = time.time()
        print("time", end-start)


        # 
        nqs = NQS(a=2., b=2., ma=1.0, mb=1.0, c=0.01, sigma=0.01)
        # generate NKBfoldsarrs data
        NKBfoldsarrs = []
        lossarr = []
        Ns = [1e5, 1e6] #,1e6]#,1e7,1e8]
        Ks = [1e5, 1e6] #,1e6]#,1e7,1e8]
        Bs = [10, 20] #, 20]
        for N in Ns:
            for K in Ks:
                for B in Bs:
                    folds = SGD(lr=0.1, momentum=0.0, nesterov=False, 
                                scheduler=None)
                    NKBfoldsarrs.append((N, K, B, folds))
                    lossarr.append(risk(N, K, B, nqs, folds))

        print("NKBfoldsarrs", NKBfoldsarrs)
        print("lossarr", lossarr)


        # test fit_multiple
        a_values = [2., 2.]#[1.5,1.8] #,1.7,1.8,1.9,2.1,2.2,2.3,2.4,2.5]
        b_values = [2.]#[0.9]#,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
        ma_values = [1.]#[1.0]#,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
        mb_values = [1.]#[1.01]#,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10]
        #b_values = [1.1, 1.3]#,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
        #nqs0_list = [NQS(a=a, b=2., ma=1.0, mb=1.0, c=0.01, sigma=0.01) for a in a_values]
        nqs0_list = []
        for a in a_values:
            for b in b_values:
                for ma in ma_values:
                    for mb in mb_values:
                        nqs0_list.append(NQS(a=a, b=b, ma=ma, mb=mb, c=0.01, sigma=0.01))
        
        #raise ValueError("nqs_length", len(nqs0_list))
    
        test_loss_computation = True
        if test_loss_computation:
            print("STARTING LOSS COMPUTATION  ")
            start = time.time()
            loss_at_nqs_list = compute_loss_multiple(NKBfoldsarrs, lossarr, nqs0_list)
            end = time.time()
            print("loss_at_nqs_list", loss_at_nqs_list)
            print("TIME to compute loss", end-start)
            #check_device(loss_at_nqs_list)
        # time the fit:
        
        test_fit = True
        if test_fit:
            print("STARTING FIT")
            start = time.time()
            fitted_nqs, trajectories = fit_multiple(NKBfoldsarrs, lossarr, nqs0_list=nqs0_list,
                                                    return_traj=True, steps=5, gtol=1e-8)
            
            end = time.time()

            print(trajectories)
            #check_device(fitted_nqs)

            print("TIME to FIT", end-start)
            print("fitted_nqs", fitted_nqs)
            #print("trajectories", trajectories)

       # if True:
            # load the trajectories
            # load the summary
       #     summary = load_trajectory_summary("nqs_trajectories")
       #     print("summary", summary)
        



