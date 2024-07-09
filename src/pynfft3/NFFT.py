#TODO: Text formatting
#TODO: Combine constructors

import warnings
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

# # allocate plan memory and init with D,N,M,n,m,f1,f2
# """
#     nfft_init(P)

# intialises the NFFT plan in C. This function does not have to be called by the user.

# # Input
# * `P` - a NFFT plan structure.

# # See also
# [`NFFT{D}`](@ref), [`nfft_finalize_plan`](@ref)

def nfft_init(P):
    D = len(P.N)

    # Convert N and n to numpy arrays for passing them to C
    Nv = np.array(P.N, dtype=np.int32)
    n = np.array(P.n, dtype=np.int32)

    # Call init for memory allocation
    ptr = nfftlib.jnfft_alloc()

    # Set the pointer
    P.plan = ctypes.cast(ptr, ctypes.POINTER(nfft_plan))

    # Initialize values
    nfftlib.jnfft_init(
        P.plan,
        ctypes.c_int32(D),
        Nv.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(P.M),
        n.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(P.m),
        ctypes.c_uint32(P.f1),
        ctypes.c_uint32(P.f2)
    )
    P.init_done = True

    #TODO: Test garbage collector
    import atexit
    atexit.register(nfft_finalize_plan, P)

def init(P):
    return nfft_init(P)

# Overwrite dot notation for plan struct in order to use C memory
def setproperty(P, v, val):
    # Init plan if not done [usually with setting nodes]
    if not P.init_done:
        nfft_init(P)

    # Prevent bad stuff from happening
    if P.finalized:
        raise ValueError("NFFT already finalized")

    # Setting nodes, verification of correct size dxM
    if v == 'x':
        D = len(P.N)
        if D == 1:
            if not isinstance(val, np.ndarray) or val.dtype != np.float64:
                raise ValueError("x has to be a Float64 vector")
            if val.shape[0] != P.M:
                raise ValueError("x has to be a Float64 vector of length M")
        else:
            if not isinstance(val, np.ndarray) or val.dtype != np.float64:
                raise ValueError("x has to be a Float64 matrix")
            if val.shape[0] != D or val.shape[1] != P.M:
                raise ValueError("x has to be a Float64 matrix of size dxM")
        
        ptr = nfftlib.jnfft_set_x(P.plan, val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        P.x = ptr

    elif v == 'f':
        if not isinstance(val, np.ndarray) or not np.issubdtype(val.dtype, np.number):
            raise ValueError("f has to be a vector of numbers")
        if val.shape[0] != P.M:
            raise ValueError("f has to be a ComplexFloat64 vector of size M")
        
        f_complex = val.astype(np.complex128)
        ptr = nfftlib.jnfft_set_f(P.plan, f_complex.ctypes.data_as(ctypes.POINTER(ctypes.c_complex)))
        P.f = ptr
        # Setting Fourier coefficients

    elif v == 'fhat':
        if not isinstance(val, np.ndarray) or not np.issubdtype(val.dtype, np.number):
            raise ValueError("fhat has to be a vector of numbers")
        l = np.prod(P.N)
        if val.shape[0] != l:
            raise ValueError("fhat has to be a ComplexFloat64 vector of size prod(N)")
        
        fhat_complex = val.astype(np.complex128)
        ptr = nfftlib.jnfft_set_fhat(P.plan, fhat_complex.ctypes.data_as(ctypes.POINTER(ctypes.c_complex)))
        P.fhat = ptr

    elif v in ['plan', 'num_threads', 'init_done', 'N', 'M', 'n', 'm', 'f1', 'f2']:
        warnings.warn(f"You can't modify {v}, please create an additional plan.")

    match v:
        case 'plan':
            warnings.warn("You can't modify the C pointer to the NFFT plan.")
        case 'num_threads':
            warnings.warn("You can't currently modify the number of threads.")
        case 'init_done':
            warnings.warn("You can't modify this flag.")
        case 'N':
            warnings.warn("You can't modify the bandwidth, please create an additional plan.")
        case 'M':
            warnings.warn("You can't modify the number of nodes, please create an additional plan.")
        case 'n':
            warnings.warn("You can't modify the oversampling parameter, please create an additional plan.")
        case 'm':
            warnings.warn("You can't modify the window size, please create an additional plan.")
        case 'f1':
            warnings.warn("You can't modify the NFFT flags, please create an additional plan.")
        case 'f2':
            warnings.warn("You can't modify the FFTW flags, please create an additional plan.")
        case _:
            setattr(P, v, val)

def setproperty(P, v):
    if v == 'x':
        if P.x is None:
            raise AttributeError("x is not set.")
        ptr = P.x
        if P.D == 1:
            return np.ctypeslib.as_array(ptr, shape=(P.M,)) # Get nodes from C memory and convert to Python type
        else:
            return np.ctypeslib.as_array(ptr, shape=(P.D, P.M)) # Get nodes from C memory and convert to Python type
    
    elif v == 'num_threads':
        return nfftlib.nfft_get_num_threads()
    
    elif v == 'f':
        if P.f is None:
            raise AttributeError("f is not set.")
        ptr = P.f
        return np.ctypeslib.as_array(ptr, shape=(P.M,)) # Get function values from C memory and convert to Python type
    
    elif v == 'fhat':
        if P.fhat is None:
            raise AttributeError("fhat is not set.")
        ptr = P.fhat
        return np.ctypeslib.as_array(ptr, shape=(np.prod(P.N),)) # Get function values from C memory and convert to Python type
    
    else:
        return P.__dict__[v]
