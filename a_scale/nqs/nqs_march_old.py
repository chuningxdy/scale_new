import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
if not (jnp.array(1.).dtype is jnp.dtype("float64")):
    raise RuntimeError("jax.config.jax_enable_x64 must be set for numerical stability. "
                      "Please set this environment variable before using NQS.")
import numpy as np
from collections import namedtuple
from functools import partial
from scipy.integrate import quad

# BASIC OBJECTS

NQS = namedtuple("NQS", ["a", "b", "m_a", "m_b", "c", "sigma"])
"""Parameters defining a Noisy Quadratic System.

Attributes:
    a (float): Decay parameter for the curvature (must be > 1)
    b (float): Decay parameter for the minimum (must be > 0)
    m_a (float): Scale parameter for the curvature (must be > 0)
    m_b (float): Scale parameter for the minimum (must be > 0)
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

def get_Q(n, nqs):
    """Compute the quadratic coefficient Q(n) for dimension n.
    
    Args:
        n (int): Dimension index
        nqs (NQS): System parameters
    
    Returns:
        float: Q(n) = m_b * n^(-b)
    """

    return nqs.m_b * n ** (-nqs.b)

def get_xstar(n, nqs):
    """Compute the optimal value x*(n) for dimension n.
    
    Args:
        n (int): Dimension index
        nqs (NQS): System parameters
    
    Returns:
        float: x*(n) = sqrt(m_a * n^(b-a) / m_b)
    """
    return jnp.sqrt(nqs.m_a * n ** (nqs.b-nqs.a) / nqs.m_b)

def get_V(n, B, nqs):
    """Compute the noise variance V(n) for dimension n.
    
    Args:
        n (int): Dimension index
        B (int): Mini-batch size
        nqs (NQS): System parameters
    
    Returns:
        float: V(n) = sigma^2 * n^(-b) / B
    """
    return nqs.sigma ** 2 * n ** (-nqs.b) / B

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

def make_schedule(K, scheduler): #folds):
    """Creates a learning rate schedule for K iterations.
    
    Args:
        K (int): Total number of iterations
        scheduler (Union[None, ConstantLR, StepLR, MultiStepLR, SequentialLR]): Learning rate scheduler
        #folds (FOLDS): Contains the scheduler specification
    
    Returns:
        tuple: (steps, factors) where:
            - steps is sequence of iteration counts for each piece
            - factors is sequence of learning rate factors for each piece
    """
    if K == 0:
        return (), ()

    match scheduler: #folds.scheduler:
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
                
                subfactors = make_schedule(Ki, subschedulers[scheduler_idx])
                # raise an error that prints subfactors
                #raise ValueError(f"subfactors: {subfactors}")
                step, factor = subfactors
                Ks.extend(step)
                factors.extend(factor)
                #Ks.extend([steps for (steps, factor) in subfactors])
                #factors.extend([factor for (steps, factor) in subfactors])
                i += 1

    return tuple(Ks), tuple(factors)

# CORE NQS RISK CALCUATIONS

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
    foldsmats = folds.mats
    Ks, factors = make_schedule(K, folds.scheduler)
    match kind:
        case 'cumu':
            return cumurisk(N, Ks, B, nqs, foldsmats, factors)
        case 'redu':
            return redurisk(N, Ks, B, nqs, foldsmats, factors)
        case 'fast':
            return fastrisk(N, Ks, B, nqs, foldsmats, factors)

@partial(jax.jit, static_argnames=("N", "Ks"))
def cumurisk(N, Ks, B, nqs, foldsmats, factors):
    """Compute cumulative risk over iterations.
    
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
    risks = jnp.cumsum(jax.vmap(dimrisk, in_axes)(Ns, Ks, B, nqs, foldsmats, factors))
    return risks + nqs.c + approx(Ns, nqs)

@partial(jax.jit, static_argnames=("Ks"))
def redurisk(N, Ks, B, nqs, foldsmats, factors):
    """Compute reduced (final) risk.
    
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
    return _redurisk_helper(N, Ks, B, nqs, foldsmats, factors) + nqs.c + approx(N, nqs)

@partial(jax.jit, static_argnames=("Ks"))
def _redurisk_helper(N, Ks, B, nqs, foldsmats, factors):
    """Compute reduced (final) risk.
    
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
    def bodyf(val):
        s, n = val
        return s + dimrisk(n, Ks, B, nqs, foldsmats, factors), n-1
    def condf(val):
        s, n = val
        return n > 0
    risk, _ = jax.lax.while_loop(condf, bodyf, (0, N))
    return risk


def fastrisk(N, Ks, B, nqs, foldsmats, factors, p=1, epsrel=1e-2):
    """Compute risk using fast quadrature approximation.
    
    Args:
        N (int): Problem dimension
        Ks (tuple): Sequence of iteration counts
        B (int): Mini-batch size
        nqs (NQS): System parameters
        foldsmats (mats): System matrices
        factors (tuple): Learning rate factors
        p (int, optional): Order of Euler-Maclaurin approximation. Must be 1 or 2. Defaults to 1.
        epsrel (float, optional): Relative error tolerance. Defaults to 1e-2.
    
    Returns:
        float: Approximated risk value
    """
    if not p in [1,2]:
        raise ValueError(f"Euler-Maclaurin order p={p} is not supported.")

    def g(n):
        return dimrisk(n, Ks, B, nqs, foldsmats, factors)

    def dgdn(n):
        return _ddr_dn(n, Ks, B, nqs, foldsmats, factors)

    def ddgdnn(n):
        return _dddr_dnn(n, Ks, B, nqs, foldsmats, factors)

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

    def abserrbound(M, scale):

        if p == 1:
            aebi = aebi1
            c_p = 2
        else:
            aebi = aebi2
            c_p = 12

        L = jnp.log(M)
        U = jnp.log(N)
        # print(f"quad of aebi on [{np.log(M)}, {np.log(N)}]")
        aeb = quad(aebi, L, U, epsabs  = c_p * epsrel * scale / 4, epsrel=0)
        # aeb = quad(aebi, L, U, epsabs  = c_p * epsrel * scale / 4, epsrel=0, full_output=True, limit=50)
        # print(f"got {aeb[0]} with nevals={aeb[2]['neval']} and subdivisions={aeb[2]['last']}")
        
        # want to return eps such that |S - hatA(m)| <= eps
        # we have that |S - hatA(m)| <= |S - A(m)| + |hatA(m) - A(m)|
        # we know that |S - A(m)| <= true(aeb) / c_p <= aeb / c_p + epsrel * scale / 4
        # we know that |hatA(m) - A(m)| <= epsrel * scale / 4

        return abs(aeb[0]) / c_p + epsrel * scale / 2

    def approxrisk(M, scale=None):

        if M > 0:
            risk = nqs.c + _redurisk_helper(M, Ks, B, nqs, foldsmats, factors)
        else:
            risk = nqs.c

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

        return risk + approx(N, nqs)
    
    vapprox = approxrisk(1)

    if abserrbound(1, abs(vapprox)) < epsrel * abs(vapprox):
        return vapprox
    
    U = N
    L = 1
    while L < U:

        M = (U+L)//2
        if  abserrbound(M, abs(vapprox)) <  epsrel * abs(vapprox):
            U = M
        else:
            L = M+1

    vapprox = approxrisk(L, scale=abs(vapprox))
    return vapprox

@jax.jit
def approx(N, nqs):
    """Compute approximation term for risk calculation.
    
    Args:
        N (int): Problem dimension
        nqs (NQS): System parameters
    
    Returns:
        float: 0.5 * m_a * zeta(a, N+1)
    """
    return 0.5 * nqs.m_a * jax.scipy.special.zeta(nqs.a, N+1)

@partial(jax.jit, static_argnames=("Ks",))
def dimrisk(n, Ks, B, nqs, foldsmats, factors):
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
    CC = jnp.kron(foldsmats.C, foldsmats.C)
    v = get_V(n, B, nqs)
    q = get_Q(n, nqs)
    var = 0
    for (K, factor) in reversed(list(zip(Ks, factors))):
        T = get_T(n, nqs, foldsmats, factor)
        TT = jnp.kron(T, T)
        Sk, TTk = superpower(TT, K)
        BB = jnp.kron(factor * foldsmats.B, factor * foldsmats.B)
        var = var + CC @ Sk @ BB
        CC = CC @ TTk
    var = 0.5 * v * q * var 
    bias = 0.5 * nqs.m_a * n ** (-nqs.a) * jnp.sum(CC)
    return jnp.squeeze(bias + var)

@partial(jax.jit, static_argnames=("Ks",))
def _ddr_dn(n, Ks, B, nqs, foldsmats, factors):
    return jax.jacfwd(dimrisk)(n, Ks, B, nqs, foldsmats, factors)

@partial(jax.jit, static_argnames=("Ks",))
def _dddr_dnn(n, Ks, B, nqs, foldsmats, factors):
    return jax.jacfwd(jax.jacfwd(dimrisk))(n, Ks, B, nqs, foldsmats, factors)

# UTILITY FUNCTIONS

@partial(jax.jit, static_argnames=('n',))
def superpower(A, n):
    """
    Compute the matrix power series sum S(A,n) and power A^n efficiently using binary decomposition.
    
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

# UTILITY FUNCTIONS

def _process_milestones(K, milestones):
    new_milestones = [0]
    i = 0
    while i < len(milestones) and milestones[i] < K:
        new_milestones.append(milestones[i])
        i += 1
    new_milestones.append(K)
    return new_milestones