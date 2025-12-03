import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
if not (jnp.array(1.).dtype is jnp.dtype("float64")):
    raise RuntimeError("jax.config.jax_enable_x64 must be set for numerical stability. "
                      "Please set this environment variable before using NQS.")
import numpy as np
from collections import namedtuple
from functools import partial
from scipy.integrate import quad
from scipy.optimize import minimize, check_grad, approx_fprime
import time
import functools
import os
import json
import datetime
import pickle



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
    A = np.array([[1.0 + momentum, -momentum],
                   [1.0, 0.0]])
    B = np.array([[-lr],
                   [0.0]])
    if nesterov:
        C = np.array([[1.0 + momentum, -momentum]])
    else:
        C = np.array([[1.0, 0.0]])
    return FOLDS(mats(A, B, C), scheduler)

#############################################
# Methods                                   #
#############################################

def get_Q(n, nqs, M = 100):
    """Compute the quadratic coefficient Q(n) for dimension n.
    
    Args:
        n (int): Dimension index
        nqs (NQS): System parameters
    
    Returns:
        float: Q(n) = mb * n^(-b)
    """
    # clip n at M, using jnp.clip
    n_clipped = jnp.clip(n, M, None)


    return nqs.mb * n_clipped ** (-nqs.b)

def get_xstar(n, nqs, M = 100):
    """Compute the optimal value x*(n) for dimension n.
    
    Args:
        n (int): Dimension index
        nqs (NQS): System parameters
    
    Returns:
        float: x*(n) = sqrt(ma * n^(b-a) / mb)
    """
    # clip n at M, using jnp.clip
    n_clipped = jnp.clip(n, M, None)

    return jnp.sqrt(nqs.ma * n_clipped ** (nqs.b-nqs.a) / nqs.mb)

def get_V(n, B, nqs, M = 100):
    """Compute the noise variance V(n) for dimension n.
    
    Args:
        n (int): Dimension index
        B (int): Mini-batch size
        nqs (NQS): System parameters
    
    Returns:
        float: V(n) = sigma^2 * n^(-b) / B
    """
    n_clipped = jnp.clip(n, M, None)

    return nqs.sigma ** 2 * n_clipped ** (-nqs.b) / B

def get_T(n, nqs, mats, factor):
    """Compute the effective transition matrix T(n) for dimension n.
    
    Args:
        n (int): Dimension index
        nqs (NQS): System parameters
        mats (mats): System matrices
        factor (float): Learning rate factor
    
    Returns:
        ndarray: T(n) = A + Q(n) * factor * B @ C
    """
    return mats.A +  get_Q(n, nqs) * factor * mats.B @ mats.C

#############################################
# Core NQS Calculations                     #
#############################################

def _core(n, Ks, B, nqs, foldsmats, factors, bias=True, var=True,
          M = 100):
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
    CC = jnp.kron(foldsmats.C, foldsmats.C)

    if var:
        vterm = 0
    else:
        vterm = None

    for (K, factor) in reversed(list(zip(Ks, factors))):
        T = get_T(n, nqs, foldsmats, factor)
        TT = jnp.kron(T, T)
        Sk, TTk = superpower(TT, K)

        if var:
            BB = jnp.kron(factor * foldsmats.B, factor * foldsmats.B)
            vterm = vterm + CC @ Sk @ BB

        CC = CC @ TTk

    if var:
        v = get_V(n, B, nqs)
        q = get_Q(n, nqs)
        vterm = jnp.squeeze(0.5 * v * q * vterm)

    if bias:
        n_clipped = jnp.clip(n, M, None)
        bterm = jnp.squeeze(0.5 * nqs.ma * n_clipped ** (-nqs.a) * jnp.sum(CC))
    else:
        bterm = None

    return bterm, vterm

def _b(n, Ks, B, nqs, foldsmats, factors):
    """Compute just the bias term for dimension n.
    
    Args:
        n (int): Dimension index
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors
    
    Returns:
        float: bias
    """
    return _core(n, Ks, B, nqs, foldsmats, factors, var=False)[0]

def _v(n, Ks, B, nqs, foldsmats, factors):
    """Compute just the variance term for dimension n
    
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
    return _core(n, Ks, B, nqs, foldsmats, factors, bias=False)[1]

def _bv(n, Ks, B, nqs, foldsmats, factors):
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

    print("compiling!!!")
    print("time:", time.time())

    bias, var = _core(n, Ks, B, nqs, foldsmats, factors)
    return bias + var

def _grada_bv(n, Ks, B, nqs, foldsmats, factors):
    """Compute the gradient of bias and variance with respect to parameter 'a'.

    Args:
        n (int): Dimension index
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors

    Returns:
        float: derivative of bias and variance terms with respect to 'a'
    """
    def f(a):
        nqs_new = nqs._replace(a=a)
        return _b(n, Ks, B, nqs_new, foldsmats, factors)

    return jax.jacfwd(f)(nqs.a)

def _gradb_bv(n, Ks, B, nqs, foldsmats, factors):
    """Compute the gradient of bias and variance with respect to parameter 'b'.

    Args:
        n (int): Dimension index
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors

    Returns:
        float: derivative of bias and variance terms with respect to 'b'
    """
    def f(b):
        nqs_new = nqs._replace(b=b)
        return _bv(n, Ks, B, nqs_new, foldsmats, factors)
        
    return jax.jacfwd(f)(nqs.b)

def _gradma_bv(n, Ks, B, nqs, foldsmats, factors):
    """Compute the gradient of bias and variance with respect to parameter 'ma'.

    Args:
        n (int): Dimension index
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors

    Returns:
        float: derivative of bias and variance terms with respect to 'ma'
    """
    nqs_new = nqs._replace(ma = 1.)
    return _b(n, Ks, B, nqs_new, foldsmats, factors)

def _gradmb_bv(n, Ks, B, nqs, foldsmats, factors):
    """Compute the gradient of bias and variance with respect to parameter 'mb'.

    Args:
        n (int): Dimension index
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors

    Returns:
        float: derivative of bias and variance terms with respect to 'mb'
    """
    def f(mb):
        nqs_new = nqs._replace(mb=mb)
        return _bv(n, Ks, B, nqs_new, foldsmats, factors)
        
    return jax.jacfwd(f)(nqs.mb)

def _gradsigma_bv(n, Ks, B, nqs, foldsmats, factors):
    """Compute the gradient of bias and variance with respect to parameter 'sigma'.

    Args:
        n (int): Dimension index
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors

    Returns:
        float: derivative of bias and variance terms with respect to 'sigma'
    """
    nqs_new = nqs._replace(sigma = jnp.sqrt(nqs.sigma))
    return 2 * jnp.sign(nqs.sigma) * _v(n, Ks, B, nqs_new, foldsmats, factors)

def _tail(N, nqs):
    """Compute approximation term and the irreducible error.
    
    Args:
        N (int): Problem dimension
        nqs (NQS): System parameters
    
    Returns:
        float: 0.5 * ma * zeta(a, N+1) + c
    """
    return 0.5 * nqs.ma * jnp.squeeze(jax.scipy.special.zeta(nqs.a, N+1)) + nqs.c


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

    def reducedf(N, Ks, B, nqs, foldsmats, factors):

        def bodyf(val):
            s, n = val
            return (s + f(n, Ks, B, nqs, foldsmats, factors), n-1)

        def condf(val):
            s, n = val
            return n > 0

        risk, _ = jax.lax.while_loop(condf, bodyf, (0, N))
        return risk

    return reducedf

def _cumurisk(N, Ks, B, nqs, foldsmats, factors):
    """Compute cumulative risk over dimensions.
    
    Args:
        N (int): Problem dimension
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors
    
    Returns:
        ndarray (len N): Cumulative risk values
    """
    Ns = jnp.arange(1, N+1)
    in_axes =  (0, None, None, None, None, None)
    risks = jnp.cumsum(jax.vmap(_bv, in_axes)(Ns, Ks, B, nqs, foldsmats, factors))
    return risks + _tail(Ns, nqs)

def _redurisk(N, Ks, B, nqs, foldsmats, factors):
    """Compute the reduced risk for the optimization problem.

    Args:
        N (int): Problem dimension
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors

    Returns:
        float: Reduced risk value
    """
    return _reducedbv(N, Ks, B, nqs, foldsmats, factors) + _tail(N, nqs)

_reducedbv = _reduce(_bv)

#############################################
# Derivatives                               #
#############################################

_dbv_dn = jax.jacfwd(_bv)
_dgrada_bv_dn = jax.jacfwd(_grada_bv)
_dgradb_bv_dn = jax.jacfwd(_gradb_bv)
_dgradma_bv_dn = jax.jacfwd(_gradma_bv)
_dgradmb_bv_dn = jax.jacfwd(_gradmb_bv)
_dgradsigma_bv_dn = jax.jacfwd(_gradsigma_bv)

_ddbv_dndn = jax.jacfwd(_dbv_dn)
_ddgrada_bv_dndn = jax.jacfwd(_dgrada_bv_dn)
_ddgradb_bv_dndn = jax.jacfwd(_dgradb_bv_dn)
_ddgradma_bv_dndn = jax.jacfwd(_dgradma_bv_dn)
_ddgradmb_bv_dndn = jax.jacfwd(_dgradmb_bv_dn)
_ddgradsigma_bv_dndn = jax.jacfwd(_dgradsigma_bv_dn)

_reducedgrada_bv = _reduce(_grada_bv)
_reducedgradb_bv = _reduce(_gradb_bv)
_reducedgradma_bv = _reduce(_gradma_bv)
_reducedgradmb_bv = _reduce(_gradmb_bv)
_reducedgradsigma_bv = _reduce(_gradsigma_bv)

_gradnqs_tail = jax.grad(_tail, 1)

#############################################
# Fast Approximate Methods                  #
#############################################

def _fastrisk(N, Ks, B, nqs, foldsmats, factors):
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
        lambda n: jax.jit(_bv, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_dbv_dn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_ddbv_dndn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda M: jax.jit(_reducedbv, static_argnames=("Ks",))(M, Ks, B, nqs, foldsmats, factors) if M > 0 else 0.,
        lambda N: jax.jit(_tail)(N, nqs)
    )

def _fastgradrisk(N, Ks, B, nqs, foldsmats, factors):
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
    da = _em(N,
        lambda n: jax.jit(_grada_bv, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_dgrada_bv_dn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_ddgrada_bv_dndn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda M: jax.jit(_reducedgrada_bv, static_argnames=("Ks",))(M, Ks, B, nqs, foldsmats, factors) if M > 0 else 0.,
        lambda N: jax.jit(_gradnqs_tail)(N, nqs).a,
    )
    db = _em(N,
        lambda n: jax.jit(_gradb_bv, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_dgradb_bv_dn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_ddgradb_bv_dndn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda M: jax.jit(_reducedgradb_bv, static_argnames=("Ks",))(M, Ks, B, nqs, foldsmats, factors) if M > 0 else 0.,
        lambda N: jax.jit(_gradnqs_tail)(N, nqs).b,
    )
    dma = _em(N,
        lambda n: jax.jit(_gradma_bv, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_dgradma_bv_dn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_ddgradma_bv_dndn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda M: jax.jit(_reducedgradma_bv, static_argnames=("Ks",))(M, Ks, B, nqs, foldsmats, factors) if M > 0 else 0.,
        lambda N: jax.jit(_gradnqs_tail)(N, nqs).ma,
    )
    dmb = _em(N,
        lambda n: jax.jit(_gradmb_bv, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_dgradmb_bv_dn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_ddgradmb_bv_dndn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda M: jax.jit(_reducedgradmb_bv, static_argnames=("Ks",))(M, Ks, B, nqs, foldsmats, factors) if M > 0 else 0.,
        lambda N: jax.jit(_gradnqs_tail)(N, nqs).mb,
    )
    dsigma = _em(N,
        lambda n: jax.jit(_gradsigma_bv, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_dgradsigma_bv_dn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda n: jax.jit(_ddgradsigma_bv_dndn, static_argnames=("Ks",))(n, Ks, B, nqs, foldsmats, factors),
        lambda M: jax.jit(_reducedgradsigma_bv, static_argnames=("Ks",))(M, Ks, B, nqs, foldsmats, factors) if M > 0 else 0.,
        lambda N: jax.jit(_gradnqs_tail)(N, nqs).sigma,
    )
    return NQS(a=da, b=db, ma=dma, mb=dmb, c=1., sigma=dsigma)

#############################################
# Wrappers                                  #
#############################################

def fold(f):
    """Adapts a function that works with expanded fold parameters to accept simplified fold inputs.
    
    This is a higher-order function that transforms a function requiring detailed fold matrices
    and schedule parameters into one that accepts a higher-level fold abstraction. It handles
    the extraction of matrices from folds and the creation of iteration schedules.
    
    Args:
        f (callable): A function with signature f(N, Ks, B, nqs, foldsmats, factors) where:
            - N: Problem dimension
            - Ks: Sequence of iteration counts (from make_schedule)
            - B: Mini-batch size
            - nqs: Neural Quadratic System parameters
            - foldsmats: System matrices extracted from folds
            - factors: Learning rate factors (from make_schedule)
    
    Returns:
        callable: A new function with signature foldf(N, K, B, nqs, folds) where:
            - N: Problem dimension
            - K: Base iteration count
            - B: Mini-batch size
            - nqs: Neural Quadratic System parameters
            - folds: Fold configuration object containing matrices
            
    Examples:
        >>> # Define a function working with low-level parameters
        >>> def _my_func(N, Ks, B, nqs, foldsmats, factors):
        ...     # Implementation
        ...     return result
        ...
        >>> # Create a version that accepts higher-level fold parameters
        >>> my_func = fold(_my_func)
        >>> # Use the adapted function with simpler parameters
        >>> result = my_func(N, K, B, nqs, folds)
    """
    def foldf(N, K, B, nqs, folds):
        foldsmats = folds.mats
        Ks, factors = make_schedule(K, folds.scheduler)
        return f(N, Ks, B, nqs, foldsmats, factors)
    return foldf

dimrisk = fold(jax.jit(_bv, static_argnames=("Ks",)))
cumurisk = fold(jax.jit(_cumurisk, static_argnames=("N", "Ks")))
redurisk = fold(jax.jit(_redurisk, static_argnames=("Ks",)))
fastrisk = fold(_fastrisk)

def risk(N, K, B, nqs, folds, kind='fast'):
    """Computes the expected risk of a FOLDS algorithm on a Noisy Quadratic System.
    
    Args:
        N (int): Dimension of the problem
        K (int): Number of iterations
        B (int): Mini-batch size
        nqs (NQS): Parameters defining the Noisy Quadratic System
        folds (FOLDS): Parameters defining the optimization algorithm
        kind (str): Type of risk calculation:
            - 'cumu': Cumulative risk over dimensions
            - 'redu': Reduced (final) risk
            - 'fast': Fast approximation using quadrature
    
    Returns:
        float or array: Expected risk value(s)
    """
    match kind:
        case 'cumu':
            return np.array(cumurisk(N, K, B, nqs, folds))
        case 'redu':
            return redurisk(N, K, B, nqs, folds)
        case 'fast':
            return fastrisk(N, K, B, nqs, folds)

redugradrisk = fold(jax.jit(jax.jacfwd(_redurisk, 3), static_argnames=("Ks",)))
fastgradrisk = fold(_fastgradrisk)

def gradrisk(N, K, B, nqs, folds, kind='fast'):
    """Computes the gradient of the expected risk of a FOLDS algorithm on a Noisy Quadratic System,
    with respect to the NQS parameters.
    
    Args:
        N (int): Dimension of the problem
        K (int): Number of iterations
        B (int): Mini-batch size
        nqs (NQS): Parameters defining the Noisy Quadratic System
        folds (FOLDS): Parameters defining the optimization algorithm
        kind (str): Type of risk calculation:
            - 'redu': Reduced (final) risk
            - 'fast': Fast approximation using quadrature
    
    Returns:
        float or array: Gradient of expected risk value(s)
    """
    match kind:
        case 'redu':
            return redugradrisk(N, K, B, nqs, folds)
        case 'fast':
            return fastgradrisk(N, K, B, nqs, folds)

#############################################
# Fitting                                   #
#############################################


def fit_fast_inprogress(NKBfoldsarr, lossarr, nqs0=None, return_res = False):

    def constant_scheduler_to_none(NKBfoldsarr):
        # if the scheduler is a constant scheduler with factior = 1.0 and total_iters = K,
        # then we can replace this with None
        # this is because the constant scheduler with factor = 1.0 is equivalent to no scheduler
        for i in range(len(NKBfoldsarr)):
            N, K, B, f = NKBfoldsarr[i]
            if isinstance(f.scheduler, ConstantLR) and f.scheduler.factor == 1.0 and f.scheduler.total_iters == K:
                NKBfoldsarr[i] = (N, K, B, FOLDS(f.mats, None))
        return NKBfoldsarr

    NKBfoldsarr = constant_scheduler_to_none(NKBfoldsarr)
    # check if the f elements of NKBfoldsarr are all identical
    fld = NKBfoldsarr[0][3]

    def folds_equal(f1, f2):
        return np.array_equal(f1.mats.A, f2.mats.A) and np.array_equal(f1.mats.B, f2.mats.B) and np.array_equal(f1.mats.C, f2.mats.C) and f1.scheduler == f2.scheduler
    
    for _, _, _, f in NKBfoldsarr:
        # check if f and fld are identical - note these are complex objects
        # so need to check the individual components
        
        if not folds_equal(f, fld):
            raise ValueError("All the folds should be identical for the NQS fitting; otherwise, use fit_general" + 
                             "\n" + str(f) + "\n" + str(fld))

        
        
    #num = len(NKBfoldsarr)
    # extract NKB from the array, and turn into a jax array
    #NKBarr = jnp.array([(N, K, B) for (N, K, B, f) in NKBfoldsarr])
    # check the dimensions of the NKB arr it should be num x 3
    #assert NKBarr.shape == (num, 3)

    def to_nqs(x):
        return NQS(a=x[0], b=x[1], ma=x[2], mb=x[3], c=x[4], sigma=x[5])

    def to_x(nqs):
        return np.array([nqs.a, nqs.b, nqs.ma, nqs.mb, nqs.c, nqs.sigma])

   # def f_old(x):
   #     nqs = to_nqs(x)
   #     return sum(0.5*(np.log(l) - np.log(risk(N, K, B, nqs, fld)))**2 / num
   #         for (N, K, B), l in zip(NKBarr, lossarr))
    
   # def df_old(x): # the log MSE version
   #     nqs = to_nqs(x)
        # parallelize this
   #     risks = np.array([risk(N, K, B, nqs, f) for (N, K, B, f) in NKBfoldsarr])
    #    drisks = np.array([gradrisk(N, K, B, nqs, f) for (N, K, B, f) in NKBfoldsarr])
    #    logrisks = np.log(risks)
    #    loglosses = np.log(lossarr)
    #    return np.sum(-(loglosses - logrisks)[:, None] * 1/risks[:, None] * drisks, axis=0) / num
    
    #def df_oldtst(x):
    #    nqs = to_nqs(x)
        
        # Extract the numeric components that vary across calls
    #    Ns = jnp.array([item[0] for item in NKBarr])  # Extract all N values
     #   Ks = jnp.array([item[1] for item in NKBarr])  # Extract all K values
    #    Bs = jnp.array([item[2] for item in NKBarr])  # Extract all B values
        # Note: We ignore the 'f' values in the tuple since we'll use fld directly
        
        # Define functions that take the varying numeric arguments
        # and use the constant objects (nqs, fld)
     #   def single_risk_vectorized(N, K, B):
    #       return risk(N, K, B, nqs, fld)
        
     #   def single_gradrisk_vectorized(N, K, B):
     #       return gradrisk(N, K, B, nqs, fld)
        
        # Vectorize these functions over the first argument (which will be mapped to N, K, B)
      #  batch_risk = jax.vmap(single_risk_vectorized)
     #   batch_gradrisk = jax.vmap(single_gradrisk_vectorized)
        
        # Apply the vectorized functions to all N, K, B values at once
      #  risks = batch_risk(Ns, Ks, Bs)
      #  drisks = batch_gradrisk(Ns, Ks, Bs)
        
      #  logrisks = jnp.log(risks)
       # loglosses = jnp.log(lossarr)
        
       # return jnp.sum(-(loglosses - logrisks)[:, None] * 1/risks[:, None] * drisks, axis=0) / num
    
    def f(x):
        nqs = to_nqs(x)
        num = len(NKBfoldsarr)
        
        # Extract data
        NKBarr_list = [(N, K, B) for (N, K, B, _) in NKBfoldsarr]
        unique_Ks = list(set(K for _, K, _ in NKBarr_list))
        
        # Initialize array for risk values
        risks = jnp.zeros(num)
        
        # Define helper function to create K-specific vectorized risk function
        def create_K_risk_function(K_val):
            @jax.jit
            def batch_risk(Ns, Bs):
                return jax.vmap(lambda N, B: risk(N, B, K_val, nqs, fld))(Ns, Bs)
            return batch_risk
        
        # Pre-compile risk functions for each unique K
        K_risk_functions = {K: create_K_risk_function(K) for K in unique_Ks}
        
        # Process for each unique K
        for K in unique_Ks:
            # Find indices where current K is used
            indices = [i for i, (_, k, _) in enumerate(NKBarr_list) if k == K]
            
            if not indices:
                continue
            
            # Extract N and B values for this K
            Ns = jnp.array([NKBarr_list[i][0] for i in indices])
            Bs = jnp.array([NKBarr_list[i][2] for i in indices])
            
            # Get the specialized risk function for this K
            batch_risk_K = K_risk_functions[K]
            
            # Compute risks for this K
            K_risks = batch_risk_K(Ns, Bs)
            
            # Update the risks array
            for idx_pos, orig_idx in enumerate(indices):
                risks = risks.at[orig_idx].set(K_risks[idx_pos])
        
        # Compute the loss using the risks
        log_risks = jnp.log(risks)
        log_losses = jnp.log(jnp.array(lossarr))
        
        # Calculate the mean squared error
        squared_errors = 0.5 * (log_losses - log_risks)**2
        return jnp.sum(squared_errors) / num

    def df(x):
        nqs = to_nqs(x)
        
        # Extract components
        NKBarr_list = [(N, K, B) for (N, K, B, _) in NKBfoldsarr]
        
        # Get unique Ks
        unique_Ks = list(set(K for _, K, _ in NKBarr_list))
        
        # Initialize arrays for results
        risks = jnp.zeros(len(NKBarr_list))
        drisks = jnp.zeros((len(NKBarr_list), len(x)))  # Assuming drisks has same dimension as x
        
        # Define risk and gradrisk functions with static K
        @functools.partial(jax.jit, static_argnames=("K",))
        def risk_with_static_K(N, B, K):
            return risk(N, B, K, nqs, fld)
        
        @functools.partial(jax.jit, static_argnames=("K",))
        def gradrisk_with_static_K(N, B, K):
            return gradrisk(N, B, K, nqs, fld)
        
        # Process for each unique K
        for K in unique_Ks:
            # Find indices where current K is used
            indices = [i for i, (_, k, _) in enumerate(NKBarr_list) if k == K]
            
            if not indices:
                continue
                
            # Extract N and B values for this K
            Ns = jnp.array([NKBarr_list[i][0] for i in indices])
            Bs = jnp.array([NKBarr_list[i][2] for i in indices])
            
            # Create vectorized functions for this specific K
            # Note: K is passed as a static value, not vectorized
            batch_risk_K = jax.vmap(lambda N, B: risk_with_static_K(N, B, K))
            batch_gradrisk_K = jax.vmap(lambda N, B: gradrisk_with_static_K(N, B, K))
            
            # Compute risks and gradients for this K
            K_risks = batch_risk_K(Ns, Bs)
            K_drisks = batch_gradrisk_K(Ns, Bs)
            
            # Update the results arrays
            for idx_pos, orig_idx in enumerate(indices):
                risks = risks.at[orig_idx].set(K_risks[idx_pos])
                drisks = drisks.at[orig_idx].set(K_drisks[idx_pos])
        
        # Compute final result
        logrisks = jnp.log(risks)
        loglosses = jnp.log(lossarr)
        
        return jnp.sum(-(loglosses - logrisks)[:, None] * 1/risks[:, None] * drisks, axis=0) / len(NKBarr_list)

    if nqs0 is None:
        nqs0 = NQS(a=2., b=1., ma=1., mb=1., c=0., sigma=1.)


    # now get the finite difference gradient at this point
    # we will use this to check the gradient using approx_fprime
    print("finite difference gradient")
    # time the finite difference gradient
    start = time.time()
    print(approx_fprime(to_x(nqs0), f, 1e-6))
    print("time taken for finite difference gradient: ", time.time() - start)

    # time the jax gradient
    print("jax gradient")
    start = time.time()
    print(df(to_x(nqs0)))
    print("time taken for jax gradient: ", time.time() - start)

    raise ValueError("stop here")



    # raise an error
    #raise ValueError("stop here")

    #minus_df = lambda x: -df(x)

    res = minimize(f, to_x(nqs0),  method='L-BFGS-B',
                   # jac = df, 
    bounds = ((1.01, None), (0.1, None), (0, None), (0, None), (0., None), (0, None)
    ))#jac=df)
    # add bounds

    
   # print(res.message)
   # print(res.status) 
   # print(res.jac)
   # print(res.fun)
    # if did not converge, raise error
    if res.status != 0:
        raise ValueError(f"NQS fit optimization failed with message: {res.message}")
    
    if return_res:
        return to_nqs(res.x), res
    else:
        return to_nqs(res.x)



def fit_new(NKBfoldsarr, lossarr, nqs0=None, 
        return_f = False, return_traj = False, 
        steps = 300, steps_per_iter = 1, gtol = 1e-3,
        return_res = False):
    num = len(NKBfoldsarr)

    norm_const = 1e5 # normalize ma, mb, c, sigma to be in the range of [-1,10]

    def to_nqs(x):

        a = x[0]
        b = x[1]
        ma = np.exp(x[2]*np.log(norm_const))
        mb = np.exp(x[3]*np.log(norm_const))
        c = np.exp(x[4]) - 1e-8
        sigma = np.exp(x[5]*np.log(norm_const)) - 1e-8

        return NQS(a=a, b=b, ma=ma, mb=mb, c=c, sigma=sigma)
    
    def to_x(nqs):

        a = nqs.a 
        b = nqs.b
        log_ma = np.log(nqs.ma)/np.log(norm_const)
        log_mb = np.log(nqs.mb)/np.log(norm_const)
        log_c = np.log(nqs.c+1e-8)
        log_sigma = np.log(nqs.sigma+1e-8)/np.log(norm_const)

        return np.array([a, b, log_ma, log_mb, log_c, log_sigma])
    

    def f(x):
        nqs = to_nqs(x)
        f_array = [(np.log(l), np.log(risk(N, K, B, nqs, f)))
            for (N, K, B, f), l in zip(NKBfoldsarr, lossarr)]
        f_value = sum(0.5*(l - r)**2 / num for l, r in f_array)
        print("f_value", f_value)


        return f_value


    
    def df(x, stage = 0, logfile = "outputs/training_log.txt"): # the log MSE version
        # create a log file
        if not os.path.exists(logfile):
            with open(logfile, "w") as filee:

                filee.write("\n\n\n\n\n\n")
                filee.write("NQS training log\n")
                filee.write("a b ma mb c sigma\n")
                
        else:
            with open(logfile, "a") as filee:
                filee.write("\n")

        nqs = to_nqs(x)
        risks = np.array([risk(N, K, B, nqs, f) for (N, K, B, f) in NKBfoldsarr])
        drisks = np.array([gradrisk(N, K, B, nqs, f) for (N, K, B, f) in NKBfoldsarr])
        logrisks = np.log(risks)
        loglosses = np.log(lossarr)
        print("nqs", nqs)
        

        #avg_risk = np.mean(risks)
        #print("avg_risk", avg_risk)
        jac_nqs = np.sum(-(loglosses - logrisks)[:, None] * 1/risks[:, None] * drisks, axis=0) / num
        jac_x = np.zeros_like(x)
        jac_x[0] = jac_nqs[0]
        jac_x[1] = jac_nqs[1]
        jac_x[2] = jac_nqs[2] * nqs.ma * np.log(norm_const)
        jac_x[3] = 0 #jac_nqs[3] * nqs.mb * np.log(norm_const) 
        jac_x[4] = jac_nqs[4] * nqs.c 
        jac_x[5] = jac_nqs[5] * nqs.sigma * np.log(norm_const)

        if stage == 1:
            jac_x[0] = 0
            jac_x[1] = 0
        elif stage == 2:
            jac_x[2] = 0
            jac_x[3] = 0
            jac_x[4] = 0
            jac_x[5] = 0
        elif stage == 3:
            jac_x[0] = 0
            jac_x[1] = 0
            jac_x[3] = 0

        print("jac_x", jac_x)
        
        with open(logfile, "a") as filee:

            filee.write(f"{nqs}\n")
            #filee.write(f"avg_risk: {avg_risk}\n")
            filee.write(f"jac_x: {jac_x}\n")
            filee.write("\n")

        return jac_x
    

    if return_f:
        return f, to_x, to_nqs

    else:
        
        if nqs0 is None:
            nqs0 = NQS(a=2., b=1., ma=1., mb=1., c=0., sigma=1.)

    
        def minimize_stage(nqs_start, stage = 0, iters = 1, gtol = 1e-4):
            
            res = minimize(f, to_x(nqs_start),  method=  'L-BFGS-B',
                            jac = lambda x: df(x, stage = stage),
                            options={'gtol': gtol, 'maxiter': iters},
            bounds = ((1.01, None), (0.1, None),
                        (None, None), 
                        (None, None),
                        (None, None), 
                        (None, None)
                ))
            return res
        

        # check if training log exists, if not create it
        if not os.path.exists("outputs/training_log.txt"):
            with open("outputs/training_log.txt", "w") as filee:
                filee.write("\n\n\n\n\n\n")
                filee.write("NQS training log\n")
                filee.write("a b ma mb c sigma\n")
                filee.write("\n")
        else:
            with open("outputs/training_log.txt", "a") as filee:
                filee.write("\n\n\n\n\n\n")
                filee.write("NQS training log\n")
                filee.write("a b ma mb c sigma\n")
                filee.write("\n")


        def minimize_alternate_sign(nqs_stt, total_iters, 
                                    itrs_per_stage = 1, gtol = 1e-4, lrr = 0.01,
                                    max_shift = 1.0, momen = 0.9,
                                    stage = 0):
            # sign gradient descent in each itr

            print("start alternating minimization")
            # write in logfile
            with open("outputs/training_log.txt", "a") as filee:
                filee.write("\n")
                filee.write("start alternating minimization\n")
                filee.write(f"total_iters: {total_iters}\n")
                filee.write(f"itrs_per_stage: {itrs_per_stage}\n")
                filee.write("\n")
            iter_pairs = total_iters // 2 // itrs_per_stage
            # list of nqs
            nqs_list = []
            nqs_list.append(nqs_stt)
            x_0 = to_x(nqs_stt)
            # initialize momentum param
            
            v = np.zeros_like(x_0)

            itr = 0
            x = to_x(nqs_stt)
            for itr in range(iter_pairs):
                # cosine schedule
                lrrr = lrr * 1/2 * (1 + np.cos(np.pi * itr / iter_pairs))

                x = to_x(nqs_stt)
               # f_value = f(x)
               # if f_value < 2.5:
               #     lrrr = lrrr * 0.01
                    # write in log
               #     with open("outputs/training_log.txt", "a") as filee:
               #         filee.write("\n")
               #         filee.write("f_value is too small, reducing lrrr\n")
               #         filee.write(f"from {lrrr} to {lrrr * 0.01}\n")
               #        filee.write("\n")
                   
                grad = df(x, stage = stage)
                # clip gradient at lrrr
                clipped_grad = np.clip(grad, -1, 1)
                
                # use momentum to update x
                v = momen * v - lrrr * clipped_grad
                x_old = x.copy()
                x = x + v
                # compute function value

                # convert x to nqs
                nqs_stt = to_nqs(x)
                nqs_list.append(nqs_stt)
                print("stage 2 nqs", nqs_stt)
               # func_value = f(x)
               # print("stage 2 function value \n", func_value)
                # add to logfile
               # with open("outputs/training_log.txt", "a") as filee:
               #     filee.write("\n")
                #    filee.write("stage 2 nqs" + str(nqs_stt) + "\n")
                #    filee.write("stage 2 function value" + str(func_value) + "\n")
                #    filee.write("\n")

                # check if converged

                # if the update magnitude is small, stop
                # the tolerance is if the inf norm of x - x_old is less than 0.01
                update_magnitude = np.linalg.norm(x - x_old, ord=np.inf)
                if update_magnitude < 0.001:
                    print("converged at step", itr * 2)
                    # write in logfile
                    with open("outputs/training_log.txt", "a") as filee:
                        filee.write("\n")
                        filee.write("converged at step" + str(itr * 2) + "\n")
                        filee.write("update magnitude" + str(update_magnitude) + " is less than 0.01\n")
                        filee.write("distance from initial guess" + str(np.linalg.norm(x - x_0)) + "\n")
                        filee.write("\n")
                    break

                if np.linalg.norm(grad) < gtol:
                    print("converged at step", itr * 2)
                    # write in logfile
                    with open("outputs/training_log.txt", "a") as filee:
                        filee.write("\n")
                        filee.write("converged at step" + str(itr * 2) + "\n")
                        filee.write("gradient norm" + str(np.linalg.norm(grad)) + " is less than gtol\n")
                        filee.write("distance from initial guess" + str(np.linalg.norm(x - x_0)) + "\n")
                        filee.write("\n")
                    break

                # check if nqs is too far from the initial guess
                # if x is too far from the initial guess, stop
                if np.linalg.norm(x - x_0) > max_shift:
                    print("x is too far from the initial guess")
                    # write in logfile
                    with open("outputs/training_log.txt", "a") as filee:
                        filee.write("\n")
                        filee.write("x is too far from the initial guess" + str(x) + "\n")
                        filee.write("distance from initial guess" + str(np.linalg.norm(x - x_0)) + "\n")
                        filee.write("\n")
                    break

            print("terminated at step", itr * 2)
            # write in logfile
            with open("outputs/training_log.txt", "a") as filee:
                filee.write("\n")
                filee.write("terminated at step" + str(itr * 2) + "\n")
                filee.write("distance from initial guess" + str(np.linalg.norm(x - x_0)) + "\n")
                filee.write("\n")
            return nqs_stt, nqs_list
        

        nqs_out1, nqs_traj1 = minimize_alternate_sign(nqs0, total_iters = steps, 
                                                    itrs_per_stage = steps_per_iter, gtol = gtol,
                                                    stage = 0,
                                                    lrr= 0.01)

        nqs_traj = nqs_traj1
        nqs_out = nqs_out1 
    
    output_dict = {}

    output_dict["nqs"] = nqs_out
    output_dict["nqs_traj"] = nqs_traj1
    output_dict["fit_metric_value"] = f(to_x(nqs_out))

    # save nqs_traj to a pickle file
    # add date time in the name
    # use the time module to get the current time


    traj_name = "outputs/nqs_traj_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
    with open(traj_name, "wb") as f:
        pickle.dump(nqs_traj, f)
    print("saved nqs_traj to", traj_name)

    return output_dict



# use this function to fit the NQS parameters when the folds are not identical in 
# the training data


def fit(NKBfoldsarr, lossarr, nqs0=None, 
        return_f = False, return_traj = False, 
        steps = 300, steps_per_iter = 1, gtol = 1e-3,
        return_res = False):
    num = len(NKBfoldsarr)

    norm_const = 1e5 # normalize ma, mb, c, sigma to be in the range of [-1,10]

    def to_nqs(x):

        a = x[0]
        b = x[1]
        ma = np.exp(x[2]*np.log(norm_const))
        mb = np.exp(x[3]*np.log(norm_const))
        c = np.exp(x[4]) - 1e-8
        sigma = np.exp(x[5]*np.log(norm_const)) - 1e-8

        return NQS(a=a, b=b, ma=ma, mb=mb, c=c, sigma=sigma)
    
    def to_x(nqs):

        a = nqs.a 
        b = nqs.b
        log_ma = np.log(nqs.ma)/np.log(norm_const)
        log_mb = np.log(nqs.mb)/np.log(norm_const)
        log_c = np.log(nqs.c+1e-8)
        log_sigma = np.log(nqs.sigma+1e-8)/np.log(norm_const)

        return np.array([a, b, log_ma, log_mb, log_c, log_sigma])
    

    def f(x):
        nqs = to_nqs(x)
        f_array = [(np.log(l), np.log(risk(N, K, B, nqs, f)))
            for (N, K, B, f), l in zip(NKBfoldsarr, lossarr)]
        f_value = sum(0.5*(l - r)**2 / num for l, r in f_array)
        print("f_value", f_value)


        return f_value


    
    def df(x, stage = 0, logfile = "outputs/training_log.txt"): # the log MSE version
        # create a log file
        if not os.path.exists(logfile):
            with open(logfile, "w") as filee:

                filee.write("\n\n\n\n\n\n")
                filee.write("NQS training log\n")
                filee.write("a b ma mb c sigma\n")
                
        else:
            with open(logfile, "a") as filee:
                filee.write("\n")

        nqs = to_nqs(x)
        risks = np.array([risk(N, K, B, nqs, f) for (N, K, B, f) in NKBfoldsarr])
        drisks = np.array([gradrisk(N, K, B, nqs, f) for (N, K, B, f) in NKBfoldsarr])
        logrisks = np.log(risks)
        loglosses = np.log(lossarr)
        print("nqs", nqs)
        

        #avg_risk = np.mean(risks)
        #print("avg_risk", avg_risk)
        jac_nqs = np.sum(-(loglosses - logrisks)[:, None] * 1/risks[:, None] * drisks, axis=0) / num
        jac_x = np.zeros_like(x)
        jac_x[0] = jac_nqs[0]
        jac_x[1] = jac_nqs[1]
        jac_x[2] = jac_nqs[2] * nqs.ma * np.log(norm_const)
        jac_x[3] = 0 #jac_nqs[3] * nqs.mb * np.log(norm_const) 
        jac_x[4] = jac_nqs[4] * nqs.c 
        jac_x[5] = jac_nqs[5] * nqs.sigma * np.log(norm_const)

        if stage == 1:
            jac_x[0] = 0
            jac_x[1] = 0
        elif stage == 2:
            jac_x[2] = 0
            jac_x[3] = 0
            jac_x[4] = 0
            jac_x[5] = 0
        elif stage == 3:
            jac_x[0] = 0
            jac_x[1] = 0
            jac_x[3] = 0

        print("jac_x", jac_x)
        
        with open(logfile, "a") as filee:

            filee.write(f"{nqs}\n")
            #filee.write(f"avg_risk: {avg_risk}\n")
            filee.write(f"jac_x: {jac_x}\n")
            filee.write("\n")

        return jac_x
    

    if return_f:
        return f, to_x, to_nqs

    else:
        
        if nqs0 is None:
            nqs0 = NQS(a=2., b=1., ma=1., mb=1., c=0., sigma=1.)

    
        def minimize_stage(nqs_start, stage = 0, iters = 1, gtol = 1e-4):
            
            res = minimize(f, to_x(nqs_start),  method=  'L-BFGS-B',
                            jac = lambda x: df(x, stage = stage),
                            options={'gtol': gtol, 'maxiter': iters},
            bounds = ((1.01, None), (0.1, None),
                        (None, None), 
                        (None, None),
                        (None, None), 
                        (None, None)
                ))
            return res
        

        # check if training log exists, if not create it
        if not os.path.exists("outputs/training_log.txt"):
            with open("outputs/training_log.txt", "w") as filee:
                filee.write("\n\n\n\n\n\n")
                filee.write("NQS training log\n")
                filee.write("a b ma mb c sigma\n")
                filee.write("\n")
        else:
            with open("outputs/training_log.txt", "a") as filee:
                filee.write("\n\n\n\n\n\n")
                filee.write("NQS training log\n")
                filee.write("a b ma mb c sigma\n")
                filee.write("\n")


        def minimize_alternate_sign(nqs_stt, total_iters, 
                                    itrs_per_stage = 1, gtol = 1e-4, lrr = 0.01,
                                    max_shift = 1.0, momen = 0.9,
                                    stage = 0):
            # sign gradient descent in each itr

            print("start alternating minimization")
            # write in logfile
            with open("outputs/training_log.txt", "a") as filee:
                filee.write("\n")
                filee.write("start alternating minimization\n")
                filee.write(f"total_iters: {total_iters}\n")
                filee.write(f"itrs_per_stage: {itrs_per_stage}\n")
                filee.write("\n")
            iter_pairs = total_iters // 2 // itrs_per_stage
            # list of nqs
            nqs_list = []
            nqs_list.append(nqs_stt)
            x_0 = to_x(nqs_stt)
            # initialize momentum param
            
            v = np.zeros_like(x_0)

            itr = 0
            x = to_x(nqs_stt)
            for itr in range(iter_pairs):
                # cosine schedule
                lrrr = lrr * 1/2 * (1 + np.cos(np.pi * itr / iter_pairs))

                x = to_x(nqs_stt)
               # f_value = f(x)
               # if f_value < 2.5:
               #     lrrr = lrrr * 0.01
                    # write in log
               #     with open("outputs/training_log.txt", "a") as filee:
               #         filee.write("\n")
               #         filee.write("f_value is too small, reducing lrrr\n")
               #         filee.write(f"from {lrrr} to {lrrr * 0.01}\n")
               #        filee.write("\n")
                   
                grad = df(x, stage = stage)
                # clip gradient at lrrr
                clipped_grad = np.clip(grad, -1, 1)
                
                # use momentum to update x
                v = momen * v - lrrr * clipped_grad
                x_old = x.copy()
                x = x + v
                # compute function value

                # convert x to nqs
                nqs_stt = to_nqs(x)
                nqs_list.append(nqs_stt)
                print("stage 2 nqs", nqs_stt)
               # func_value = f(x)
               # print("stage 2 function value \n", func_value)
                # add to logfile
               # with open("outputs/training_log.txt", "a") as filee:
               #     filee.write("\n")
                #    filee.write("stage 2 nqs" + str(nqs_stt) + "\n")
                #    filee.write("stage 2 function value" + str(func_value) + "\n")
                #    filee.write("\n")

                # check if converged

                # if the update magnitude is small, stop
                # the tolerance is if the inf norm of x - x_old is less than 0.01
                update_magnitude = np.linalg.norm(x - x_old, ord=np.inf)
                if update_magnitude < 0.001:
                    print("converged at step", itr * 2)
                    # write in logfile
                    with open("outputs/training_log.txt", "a") as filee:
                        filee.write("\n")
                        filee.write("converged at step" + str(itr * 2) + "\n")
                        filee.write("update magnitude" + str(update_magnitude) + " is less than 0.01\n")
                        filee.write("distance from initial guess" + str(np.linalg.norm(x - x_0)) + "\n")
                        filee.write("\n")
                    break

                if np.linalg.norm(grad) < gtol:
                    print("converged at step", itr * 2)
                    # write in logfile
                    with open("outputs/training_log.txt", "a") as filee:
                        filee.write("\n")
                        filee.write("converged at step" + str(itr * 2) + "\n")
                        filee.write("gradient norm" + str(np.linalg.norm(grad)) + " is less than gtol\n")
                        filee.write("distance from initial guess" + str(np.linalg.norm(x - x_0)) + "\n")
                        filee.write("\n")
                    break

                # check if nqs is too far from the initial guess
                # if x is too far from the initial guess, stop
                if np.linalg.norm(x - x_0) > max_shift:
                    print("x is too far from the initial guess")
                    # write in logfile
                    with open("outputs/training_log.txt", "a") as filee:
                        filee.write("\n")
                        filee.write("x is too far from the initial guess" + str(x) + "\n")
                        filee.write("distance from initial guess" + str(np.linalg.norm(x - x_0)) + "\n")
                        filee.write("\n")
                    break

            print("terminated at step", itr * 2)
            # write in logfile
            with open("outputs/training_log.txt", "a") as filee:
                filee.write("\n")
                filee.write("terminated at step" + str(itr * 2) + "\n")
                filee.write("distance from initial guess" + str(np.linalg.norm(x - x_0)) + "\n")
                filee.write("\n")
            return nqs_stt, nqs_list
        

        nqs_out1, nqs_traj1 = minimize_alternate_sign(nqs0, total_iters = steps, 
                                                    itrs_per_stage = steps_per_iter, gtol = gtol,
                                                    stage = 0,
                                                    lrr= 0.01)

        nqs_traj = nqs_traj1
        nqs_out = nqs_out1 
    
    output_dict = {}

    output_dict["nqs"] = nqs_out
    output_dict["nqs_traj"] = nqs_traj1
    output_dict["fit_metric_value"] = f(to_x(nqs_out))

    # save nqs_traj to a pickle file
    # add date time in the name
    # use the time module to get the current time


    traj_name = "outputs/nqs_traj_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
    with open(traj_name, "wb") as f:
        pickle.dump(nqs_traj, f)
    print("saved nqs_traj to", traj_name)

    return output_dict



def fit_backupp(NKBfoldsarr, lossarr, nqs0=None, return_res = False):
    num = len(NKBfoldsarr)

    #def to_nqs(x):
    #    return NQS(a=x[0], b=x[1], ma=x[2], mb=x[3], c=x[4], sigma=x[5])

    #def to_x(nqs):
    #    return np.array([nqs.a, nqs.b, nqs.ma, nqs.mb, nqs.c, nqs.sigma])

    def to_nqs(x):
        a = x[0]
        b = x[1]
        ma = np.exp(x[2])*np.log(100000)
        mb = np.exp(x[3])*np.log(100000)
        c = np.exp(x[4]) - 1e-8
        sigma = np.exp(x[5]) - 1e-8

        #return NQS(a=x[0], b=x[1], ma=x[2], mb=x[3], c=x[4], sigma=x[5])
        return NQS(a=a, b=b, ma=ma, mb=mb, c=c, sigma=sigma)
    
    def to_x(nqs):
        a = nqs.a 
        b = nqs.b
        log_ma = np.log(nqs.ma)/np.log(100000)
        log_mb = np.log(nqs.mb)/np.log(100000)
        log_c = np.log(nqs.c+1e-8)
        log_sigma = np.log(nqs.sigma+1e-8)
        #
        #return np.array([nqs.a, nqs.b, nqs.ma, nqs.mb, nqs.c, nqs.sigma])
        #print(nqs, "nqs")
        #print(a, b, log_ma, log_mb, log_c, log_sigma)
        #raise ValueError("stop here")
        return np.array([a, b, log_ma, log_mb, log_c, log_sigma])

    def f(x):
        nqs = to_nqs(x)
        return sum(0.5*(np.log(l) - np.log(risk(N, K, B, nqs, f)))**2 / num
            for (N, K, B, f), l in zip(NKBfoldsarr, lossarr))

    def df_old(x):
        nqs = to_nqs(x)
        return sum(-(l - risk(N, K, B, nqs, f)) * to_x(gradrisk(N, K, B, nqs, f)) / num
            for (N, K, B, f), l in zip(NKBfoldsarr, lossarr))
    
    def df_zero(x): # all zeros
        nqs = to_nqs(x)
        return np.zeros_like(x)
    
    
    def df(x, log_file = "outputs/df_log.txt"): # the log MSE version
        if not os.path.exists(log_file):
            with open(log_file, "w") as filee:
                # add space
                filee.write("\n\n\n\n\n")
                # write header
                filee.write("nqs, jac_x\n")
                
        nqs = to_nqs(x)
        # parallelize this
        risks = np.array([risk(N, K, B, nqs, f) for (N, K, B, f) in NKBfoldsarr])
        risks = risks 
        drisks = np.array([gradrisk(N, K, B, nqs, f) for (N, K, B, f) in NKBfoldsarr])
        logrisks = np.log(risks)
        #loglosses = np.log(lossarr - lower_l)
        losses = np.array(lossarr) 
        loglosses = np.log(losses)
        #return np.sum(-(loglosses - logrisks)[:, None] * 1/risks[:, None] * drisks, axis=0) / num
        print("nqs", nqs)
       # avg_risk = np.mean(risks)
       # print("avg_risk", avg_risk)
        #jac_x = np.sum(-(loglosses - logrisks)[:, None] * 1/risks[:, None] * drisks, axis=0) / num
        jac_nqs = np.sum(-(loglosses - logrisks)[:, None] * 1/risks[:, None] * drisks, axis=0) / num
        jac_x = np.zeros_like(x)
        jac_x[0] = 0 #jac_nqs[0]
        jac_x[1] = 0 #jac_nqs[1]
        jac_x[2] = jac_nqs[2] * nqs.ma * np.log(1000000)
        jac_x[3] = 0 #jac_nqs[3] * nqs.mb * np.log(100000) 
        jac_x[4] = jac_nqs[4] * nqs.c
        jac_x[5] = jac_nqs[5] * nqs.sigma 
        print("jac_x", jac_x)
        # write nqs, avg_risk, jac_x to log file
        # if log_file does not exist, create it

        # append to the log file
        with open(log_file, "a") as filee:
            filee.write(f"{nqs}\n")
            #filee.write(f"{avg_risk}\n")
            filee.write(f"x: {x}\n")
            filee.write(f"{jac_x}\n")

            filee.write("\n\n")
        return jac_x


    def df_new_tst(x):
        nqs = to_nqs(x)
        
        # Define single-item functions to be vectorized
        def single_risk(args):
            N, K, B, f = args
            return risk(N, K, B, nqs, f)
        
        def single_gradrisk(args):
            N, K, B, f = args
            return gradrisk(N, K, B, nqs, f)
        
        # Vectorize the functions using vmap
        batch_risk = jax.vmap(single_risk)
        batch_gradrisk = jax.vmap(single_gradrisk)
        
        # Apply the vectorized functions to the entire array at once
        risks = batch_risk(NKBfoldsarr)
        drisks = batch_gradrisk(NKBfoldsarr)
        
        logrisks = jnp.log(risks)
        loglosses = jnp.log(lossarr)
        
        return jnp.sum(-(loglosses - logrisks)[:, None] * 1/risks[:, None] * drisks, axis=0) / num
    

    if nqs0 is None:
        nqs0 = NQS(a=2., b=1., ma=1., mb=1., c=0., sigma=1.)

    #minus_df = lambda x: -df(x)

    max_eps = jnp.log(2.5) #2.5)
    max_log_mb = jnp.log(100) #/np.log(1e5)
    res = minimize(f, to_x(nqs0),  method='L-BFGS-B',
                    jac = df, 
                    options={'maxiter': 30, 'gtol': 1e-3},
    bounds = ((1.01, None), (0.1, None), 
              (None, None), 
              (None, max_log_mb),
              (None, max_eps), 
              (None, None)
    ))

    
   # print(res.message)
   # print(res.status) 
   # print(res.jac)
   # print(res.fun)
    # if did not converge, raise error
 #   if res.status != 0:
 #       raise ValueError(f"NQS fit optimization failed with message: {res.message}")
    
    if return_res:
        return to_nqs(res.x), res
    else:
        return to_nqs(res.x)

#############################################
# Compute-optimal allocation                #
#############################################

# TODO

#############################################
# Utilities                                 #
#############################################

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


def _em(N, g, dgdn, ddgdnn, headsum, tailsum, p=1, epsrel=1e-2):
    """Euler-Maclaurin approximation for numerical integration in NQS calculations.

    This function applies the Euler-Maclaurin formula to approximate sums in NQS risk calculations,
    providing faster code compared to direct summation at the cost of epsrel approximation error.

    Args:
        N (int): Upper limit of summation for the "head" part
        g (callable): Function to sum
        dgdn (callable): First derivative of g in n
        ddgdnn (callable): Second derivative of g in n
        headsum (float): Sum for the "head" part (usually computed directly)
        tailsum (float): Sum for the "tail" part (usually an analytical expression)
        p (int, optional): Order of approximation. Defaults to 1. Must be 1 or 2.
        epsrel (float, optional): Relative error tolerance. Defaults to 1e-2.

    Returns:
        float: Approximated sum
    """
    if not p in [1,2]:
        raise ValueError(f"Euler-Maclaurin order p={p} is not supported.")

    def f(x):
        n = np.exp(x)
        dn = np.exp(x)
        return g(n) * dn

    def aebi1(x):
        n = np.exp(x)
        dn = np.exp(x)
        return abs(dgdn(n)) * dn

    def aebi2(x):
        n = np.exp(x)
        dn = np.exp(x)
        return abs(ddgdnn(n)) * dn

    def err(M, scale):

        if p == 1:
            aebi = aebi1
            c_p = 2
        else:
            aebi = aebi2
            c_p = 12

        L = jnp.log(M)
        U = jnp.log(N)
        # print(f"quad of aebi on [{np.log(M)}, {np.log(N)}]")
        aeb = quad(aebi, L, U, epsabs  = c_p * epsrel * scale / 4, epsrel=1e-8) # 1e-14 is required by quad ()> 50* machine eps for float64
        # aeb = quad(aebi, L, U, epsabs  = c_p * epsrel * scale / 4, epsrel=0, full_output=True, limit=50)
        # print(f"got {aeb[0]} with nevals={aeb[2]['neval']} and subdivisions={aeb[2]['last']}")
        
        # want to return eps such that |S - hatA(m)| <= eps
        # we have that |S - hatA(m)| <= |S - A(m)| + |hatA(m) - A(m)|
        # we know that |S - A(m)| <= true(aeb) / c_p <= aeb / c_p + epsrel * scale / 4
        # we know that |hatA(m) - A(m)| <= epsrel * scale / 4

        return abs(aeb[0]) / c_p + epsrel * scale / 2

    def vapprox(M, scale=None):

        risk = headsum(M)
        L = jnp.log(M)
        U = jnp.log(N)
        # print(f"quad of f on [{np.log(M)}, {np.log(N)}]")
        if scale:
            integral = quad(f, L, U, epsabs = epsrel * scale / 4, epsrel=0)
        else:
            integral = quad(f, L, U, epsrel = epsrel, epsabs=0)
        # print(f"got {integral[0]} with nevals={integral[2]['neval']} and subdivisions={integral[2]['last']}")

        risk += integral[0]
        risk += (g(N)-g(M)) / 2

        if p == 2:
            risk += (dgdn(float(N)) - dgdn(float(M))) / 12

        return risk + tailsum(N)
    
    v = vapprox(1)

    if err(1, abs(v)) < epsrel * abs(v):
        return v
    
    U = N
    L = 1
    while L < U:

        M = (U+L)//2
        if  err(M, abs(v)) <  epsrel * abs(v):
            U = M
        else:
            L = M+1

    v = vapprox(L, scale=abs(v))
    return v

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