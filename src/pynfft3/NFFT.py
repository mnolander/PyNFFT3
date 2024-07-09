#TODO: Text formatting

import ctypes
import numpy as np
from .flags import *
from . import nfftlib

# """
# NFFT{D}

# A NFFT (nonequispaced fast Fourier transform) plan, where D is the dimension. 

# Considering a D-dimensional trigonometric polynomial

# ```math
# f colon mathbb{T}^D \to mathbb{C}, \; f(\pmb{x}) colon = \sum_{\pmb{k} \in I_{\pmb{N}}^D} \hat{f}_{\pmb{k}} \, mathrm{e}^{-2 \pi mathrm{i} \, \pmb{k} cdot \pmb{x}}
# ```

# with an index set ``I_{\pmb{N}}^D coloneqq \left\{ \pmb{k} \in mathbb{Z}^D: - \frac{N_i}{2} \leq k_i \leq \frac{N_i}{2} - 1, \, i = 1,2,\ldots,D \right\}`` where ``\pmb{N} \in (2mathbb{N})^{D}`` is the multibandlimit. 
# The NDFT (nonequispaced discrete Fourier transform) is its evaluation at ``M \in mathbb{N}`` arbitrary points ``\pmb{x}_j \in [-0.5,0.5)^D`` for ``j = 1, \ldots, M``,

# ```math
# f(\pmb{x}_j) colon = \sum_{\pmb{k} \in I^D_{\pmb{N}}} \hat{f}_{\pmb{k}} \, mathrm{e}^{-2 \pi mathrm{i} \, \pmb{k} cdot \pmb{x}_j}
# ```

# with given coefficients ``\hat{f}_{\pmb{k}} \in mathbb{C}``. The NFFT is an algorithm for the fast evaluation of the NDFT and the adjoint problem, the fast evaluation of the adjoint NDFT

# ```math
# \hat{h}_{\pmb{k}} coloneqq \sum^{M}_{j = 1} f_j \, mathrm{e}^{-2 \pi mathrm{i} \, pmb{k} cdot pmb{x}_j}, \, pmb{k} \in I_{pmb{N}}^D,
# ```

# for given coefficients ``f_j \in mathbb{C}, j =1,2,\ldots,M``. Note that in general, the adjoint NDFT is not the inverse transform of the NDFT.

# # Fields
# * `N` - the multibandlimit ``(N_1, N_2, \ldots, N_D)`` of the trigonometric polynomial ``f``.
# * `M` - the number of nodes.
# * `n` - the oversampling ``(n_1, n_2, \ldots, n_D)`` per dimension.
# * `m` - the window size. A larger m results in more accuracy but also a higher computational cost. 
# * `f1` - the NFFT flags.
# * `f2` - the FFTW flags.
# * `init_done` - indicates if the plan is initialized.
# * `finalized` - indicates if the plan is finalized.
# * `x` - the nodes ``pmb{x}_j \in [-0.5,0.5)^D, j = 1, \ldots, M``.
# * `f` - the values ``f(pmb{x}_j)`` for the NFFT or the coefficients ``f_j \in mathbb{C}, j = 1, \ldots, M,`` for the adjoint NFFT.
# * `fhat` - the Fourier coefficients ``\hat{f}_{pmb{k}} \in mathbb{C}`` for the NFFT or the values ``\hat{h}_{pmb{k}}, pmb{k} \in I_{\pmb{N}}^D,`` for the adjoint NFFT.
# * `plan` - plan (C pointer).

# # Constructor
#     NFFT{D}( N::NTuple{D,Int32}, M::Int32, n::NTuple{D,Int32}, m::Int32, f1::UInt32, f2::UInt32 ) where D

# # Additional Constructor
#     NFFT( N::NTuple{D,Int32}, M::Int32, n::NTuple{D,Int32}, m::Int32, f1::UInt32, f2::UInt32 ) where {D}
#     NFFT( N::NTuple{D,Int32}, M::Int32 ) where {D}
# """

# Create class for NFFT plan
class nfft_plan(ctypes.Structure):
    pass

class NFFT:
    def __init__(self, N, M, n, m, f1, f2):
        if any(x <= 0 for x in N):
            raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")
        
        if sum(x % 2 for x in N) != 0:
            raise ValueError(f"Invalid N: {N}. Argument must be an even integer")
        
        if M <= 0:
            raise ValueError(f"Invalid M: {M}. Argument must be a positive integer")
        
        if any(x <= 0 for x in n):
            raise ValueError(f"Invalid n: {n}. Argument must be a positive integer")
        
        if any(x <= y for x, y in zip(n, N)):
            raise ValueError(f"Invalid n: {n}. Argument must fulfill n_i > N_i")
        
        if sum(x % 2 for x in n) != 0:
            raise ValueError(f"Invalid n: {n}. Argument must be an even integer")
        
        if m <= 0:
            raise ValueError(f"Invalid m: {m}. Argument must be a positive integer")
        
        self.N = N  # bandwidth tuple
        self.M = M  # number of nodes
        self.n = n  # oversampling per dimension
        self.m = m  # window size
        self.f1 = f1  # NFFT flags
        self.f2 = f2  # FFTW flags
        self.init_done = False  # bool for plan init
        self.finalized = False  # bool for finalizer
        self.x = None  # nodes, will be set later
        self.f = None  # function values
        self.fhat = None  # Fourier coefficients
        self.plan = ctypes.POINTER(nfft_plan)()  # plan (C pointer)

# additional constructor for easy use [NFFT((N,N),M) instead of NFFT{2}((N,N),M)]
def create_NFFT(N, M):
    if any(x <= 0 for x in N):
        raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")

    # Convert N to a vector for passing to C
    Nv = np.array(N, dtype=np.int32)

    # Default oversampling
    n = tuple((2 ** (np.ceil(np.log(Nv) / np.log(2)) + 1)).astype(np.int32))

    # Default NFFT flags
    D = len(N)
    if D > 1:
        f1 = f1_default
    else:
        f1 = f1_default_1d

    return NFFT(N, M, n, 8, f1, f2_default)

def create_NFFT_with_defaults(N, M, n, m=int(default_window_cut_off), f1=None, f2=f2_default):
    D = len(N)

    if any(x <= 0 for x in N):
        raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")

    if any(x <= 0 for x in n):
        raise ValueError(f"Invalid n: {n}. Argument must be a positive integer")

    if f1 is None:
        f1 = f1_default if D > 1 else f1_default_1d

    # Additional flags
    f1 |= (MALLOC_X | MALLOC_F_HAT | MALLOC_F | FFTW_INIT)

    return NFFT(N, M, n, m, f1, f2)

# # finalizer
# """
#     nfft_finalize_plan(P)

# destroys a NFFT plan structure.

# # Input
# * `P` - a NFFT plan structure.

# # See also
# [`NFFT{D}`](@ref), [`nfft_init`](@ref)
# """

def nfft_finalize_plan(P):
    if not P.init_done:
        raise ValueError("NFFT not initialized.")

    if not P.finalized:
        P.finalized = True
        nfftlib.jnfft_finalize(ctypes.byref(P.plan))

def finalize_plan(P):
    return nfft_finalize_plan(P)