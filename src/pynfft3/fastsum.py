import warnings
import ctypes
import numpy as np
import atexit
from .flags import *
from . import fastsumlib
from . import fastsum_plan

# """
#     FASTSUM

# The fast summation algorithm evaluates the function

# ```math
# f(\pmb{y}) \coloneqq \sum^{N}_{k = 1} \alpha_k \, \mathscr{K}(\pmb{y} - \pmb{x}_k) = \sum^{N}_{k = 1} \alpha_k \, K(\lVert \pmb{y} - \pmb{x}_k \rVert_2)
# ```

# for given arbitrary source knots ``\pmb{x}_k \in \mathbb{R}^d, k = 1,2, \cdots, N`` and a given kernel function ``\mathscr{K}(\cdot) = K (\lVert \cdot \rVert_2), \; \pmb{x} \in \mathbb{R}^d``, 
# which is an even, real univariate function which is infinitely differentiable at least in ``\mathbb{R} \setminus \{ 0 \}``. 
# If ``K`` is infinitely differentiable at zero as well, then ``\mathscr{K}`` is defined on ``\mathbb{R}^d`` and is called 
# nonsingular kernel function. The evaluation is done at ``M`` different points ``\pmb{y}_j \in \mathbb{R}^d, j = 1, \cdots, M``. 

# # Fields
# * `d` - dimension.
# * `N` - number of source nodes.
# * `M` - number of target nodes.
# * `n` - expansion degree.
# * `p` - degree of smoothness.
# * `kernel` - name of kernel function ``K``.
# * `c` - kernel parameters.
# * `eps_I` - inner boundary.
# * `eps_B` - outer boundary.
# * `nn_x` - oversampled nn in x.
# * `nn_y` - oversampled nn in y.
# * `m_x` - NFFT-cutoff in x.
# * `m_y` - NFFT-cutoff in y.
# * `init_done` - bool for plan init.
# * `finalized` - bool for finalizer.
# * `flags` - flags.
# * `x` - source nodes.
# * `y` - target nodes.
# * `alpha` - source coefficients.
# * `f` - target evaluations.
# * `plan` - plan (C pointer).

# # Constructor
#     FASTSUM( d::Integer, N::Integer, M::Integer, n::Integer, p::Integer, kernel::String, c::Vector{<:Real}, eps_I::Real, eps_B::Real, nn_x::Integer, nn_y::Integer, m_x::Integer, m_y::Integer, flags::UInt32 )

# # Additional Constructor
#     FASTSUM( d::Integer, N::Integer, M::Integer, n::Integer, p::Integer, kernel::String, c::Real, eps_I::Real, eps_B::Real, nn::Integer, m::Integer )

# # See also
# [`NFFT`](@ref)
# """

# Set arugment and return types for functions
nfstlib.jnfst_init.argtypes = [ctypes.POINTER(nfst_plan), 
                               ctypes.c_int32, 
                               ctypes.POINTER(ctypes.c_int32), 
                               ctypes.c_int32, 
                               ctypes.POINTER(ctypes.c_int32), 
                               ctypes.c_int32, 
                               ctypes.c_uint32, 
                               ctypes.c_uint32]

nfstlib.jnfst_alloc.restype = ctypes.POINTER(nfst_plan)
nfstlib.jnfst_finalize.argtypes = [ctypes.POINTER(nfst_plan)]

nfstlib.jnfst_set_x.argtypes = [ctypes.POINTER(nfst_plan), np.ctypeslib.ndpointer(np.float64, flags='C')]
nfstlib.jnfst_set_f.argtypes = [ctypes.POINTER(nfst_plan), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C')] 
nfstlib.jnfst_set_fhat.argtypes = [ctypes.POINTER(nfst_plan), np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C')] 

nfstlib.jnfst_trafo.argtypes = [ctypes.POINTER(nfst_plan)]
nfstlib.jnfst_adjoint.argtypes = [ctypes.POINTER(nfst_plan)]
nfstlib.jnfst_trafo_direct.argtypes = [ctypes.POINTER(nfst_plan)]
nfstlib.jnfst_adjoint_direct.argtypes = [ctypes.POINTER(nfst_plan)]

class FASTSUM:
    def __init__(self, d, N, M, kernel, c, n=256, p=8, eps_I=256/8, eps_B=1/16, nn=512, m=8):
        self.plan = fastsumlib.jfastsum_alloc()
        
        if N <= 0:
            raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")
        if M <= 0:
            raise ValueError(f"Invalid M: {M}. Argument must be a positive integer")
        if n <= 0:
            raise ValueError(f"Invalid n: {n}. Argument must be a positive integer")
        if m <= 0:
            raise ValueError(f"Invalid m: {m}. Argument must be a positive integer")

        self.d = d  # dimension
        self.N = N  # number of source nodes
        self.M = M  # number of target nodes
        self.n = n  # expansion degree
        self.p = p  # degree of smoothness
        self.kernel = kernel  # name of kernel
        self.c = np.array([c], dtype=np.float64) if isinstance(c, (int, float)) else np.array(c, dtype=np.float64)  # kernel parameters
        self.eps_I = eps_I if eps_I is not None else 256 / 8  # inner boundary
        self.eps_B = eps_B  # outer boundary
        self.nn_x = nn  # oversampled nn in x
        self.nn_y = nn  # oversampled nn in y
        self.m_x = m  # NFFT-cutoff in x
        self.m_y = m  # NFFT-cutoff in y
        fastsumlib.jfastsum_init(self.plan)
        self.init_done = True  # bool for plan init
        self.finalized = False  # bool for finalizer
        self.flags = 0  # flags
        self.x = None  # source nodes
        self.y = None  # target nodes
        self.alpha = None  # source coefficients
        self.f = None  # target evaluations
    
    def fastsum_finalize_plan(self):
        fastsumlib.jfastsum_finalize.argtypes = (
            ctypes.POINTER(fastsum_plan),   # P
        )

        if not self.init_done:
            raise ValueError("FASTSUM not initialized.")

        if not self.finalized:
            self.finalized = True
            fastsumlib.jfastsum_finalize(self.plan)

    def finalize_plan(self):
        return self.fastsum_finalize_plan()