import warnings
import ctypes
import numpy as np
import atexit
from .flags import *
from . import nfftlib
from . import nfft_plan

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

nfftlib.jnfft_init.argtypes = [ctypes.POINTER(nfft_plan), 
                               ctypes.c_int32, 
                               ctypes.POINTER(ctypes.c_int32), 
                               ctypes.c_int32, 
                               ctypes.POINTER(ctypes.c_int32), 
                               ctypes.c_int32, 
                               ctypes.c_uint32, 
                               ctypes.c_uint32]

nfftlib.jnfft_alloc.restype = ctypes.POINTER(nfft_plan)

class NFFT:
    def __init__(self, N, M, n=None, m=default_window_cut_off, f1=None, f2=f2_default):
        self.plan = nfftlib.jnfft_alloc()
        self.N = N  # bandwidth tuple
        N_ct = N.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self.M = M  # number of nodes
        self.n = n  # oversampling per dimension
        self.m = m  # window size
        self.D = len(N)  # dimensions

        if any(x <= 0 for x in N):
            raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")
        
        if sum(x % 2 for x in N) != 0:
            raise ValueError(f"Invalid N: {N}. Argument must be an even integer")
        
        if M <= 0:
            raise ValueError(f"Invalid M: {M}. Argument must be a positive integer")

        if n is None:
            self.n = (2 ** (np.ceil( np.log(self.N) / np.log(2)) +1 )).astype('int32')
            n_ct = self.n.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        if any(x <= 0 for x in self.n):
            raise ValueError(f"Invalid n: {self.n}. Argument must be a positive integer")
        
        if any(x <= y for x, y in zip(self.n, N)):
            raise ValueError(f"Invalid n: {self.n}. Argument must fulfil n_i > N_i")
        
        if sum(x % 2 for x in self.n) != 0:
            raise ValueError(f"Invalid n: {self.n}. Argument must be an even integer")
        
        if m <= 0:
            raise ValueError(f"Invalid m: {m}. Argument must be a positive integer")

        if f1 is None:
            self.f1 = f1_default if self.D > 1 else f1_default_1d
        else:
            self.f1 = f1

        self.f2 = f2  # FFTW flags
        nfftlib.jnfft_init(self.plan, self.D, N_ct, self.M, n_ct, self.m, self.f1, self.f2)
        self.init_done = False  # bool for plan init
        self.finalized = False  # bool for finalizer
        self.x = None  # nodes, will be set later
        self.f = None  # function values
        self.fhat = None  # Fourier coefficients
    
    # # finalizer
    # """
    #     nfft_finalize_plan(P)

    # destroys a NFFT plan structure.

    # # Input
    # * `P` - a NFFT plan structure.

    # # See also
    # [`NFFT{D}`](@ref), [`nfft_init`](@ref)
    # """

    def nfft_finalize_plan(self):
        nfftlib.jnfft_finalize.argtypes = (
            ctypes.POINTER(nfft_plan),   # P
        )

        if not self.init_done:
            raise ValueError("NFFT not initialized.")

        if not self.finalized:
            self.finalized = True
            nfftlib.jnfft_finalize(self.plan)

    def finalize_plan(self):
        return self.nfft_finalize_plan()

    # # allocate plan memory and init with D,N,M,n,m,f1,f2
    # """
    #     nfft_init(P)

    # intialises the NFFT plan in C. This function does not have to be called by the user.

    # # Input
    # * `P` - a NFFT plan structure.

    # # See also
    # [`NFFT{D}`](@ref), [`nfft_finalize_plan`](@ref)

    def nfft_init(self):
        # Convert N and n to numpy arrays for passing them to C
        Nv = np.array(self.N, dtype=np.int32)
        n = np.array(self.n, dtype=np.int32)

        # Call init for memory allocation
        nfftlib.jnfft_alloc.restype = ctypes.POINTER(nfft_plan)
        ptr = nfftlib.jnfft_alloc()

        # Set the pointer
        self.plan = ctypes.cast(ptr, ctypes.POINTER(nfft_plan))

        nfftlib.jnfft_init.argtypes = (
            ctypes.POINTER(nfft_plan),   # P
            ctypes.c_int,               # D
            ctypes.POINTER(ctypes.c_int), # N
            ctypes.c_int,               # M
            ctypes.POINTER(ctypes.c_int), # n
            ctypes.c_int,               # m
            ctypes.c_uint,              # f1
            ctypes.c_uint               # f2
        )

        # Initialize values
        nfftlib.jnfft_init(
            self.plan,
            ctypes.c_int(self.D),
            ctypes.cast(Nv.ctypes.data, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(self.M),
            ctypes.cast(n.ctypes.data, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(self.m),
            ctypes.c_uint(self.f1),
            ctypes.c_uint(self.f2)
        )
        self.init_done = True

        atexit.register(self.nfft_finalize_plan())

    def init(self):
        return self.nfft_init()

    # Overwrite dot notation for plan struct in order to use C memory
    def setproperty(self, v, val):
        # Init plan if not done [usually with setting nodes]
        if not self.init_done:
            self.nfft_init

        # Prevent bad stuff from happening
        if self.finalized:
            raise ValueError("NFFT already finalized")

        # Setting nodes, verification of correct size dxM
        if v == 'x':
            if self.D == 1:
                if not isinstance(val, np.ndarray) or val.dtype != np.float64:
                    raise ValueError("x has to be a Float64 vector")
                if val.shape[0] != self.M:
                    raise ValueError("x has to be a Float64 vector of length M")
            else:
                if not isinstance(val, np.ndarray) or val.dtype != np.float64:
                    raise ValueError("x has to be a Float64 matrix")
                if val.shape[0] != self.D or val.shape[1] != self.M:
                    raise ValueError("x has to be a Float64 matrix of size dxM")
            
            ptr = nfftlib.jnfft_set_x(self.plan, val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            self.x = ptr

        elif v == 'f':
            if not isinstance(val, np.ndarray) or not np.issubdtype(val.dtype, np.number):
                raise ValueError("f has to be a vector of numbers")
            if val.shape[0] != self.M:
                raise ValueError("f has to be a ComplexFloat64 vector of size M")
            
            f_complex = val.astype(np.complex128)
            ptr = nfftlib.jnfft_set_f(self.plan, f_complex.ctypes.data_as(ctypes.POINTER(np.complex128)))
            self.f = ptr
            # Setting Fourier coefficients

        elif v == 'fhat':
            if not isinstance(val, np.ndarray) or not np.issubdtype(val.dtype, np.number):
                raise ValueError("fhat has to be a vector of numbers")
            l = np.prod(self.N)
            if val.shape[0] != l:
                raise ValueError("fhat has to be a ComplexFloat64 vector of size prod(N)")
            
            fhat_complex = val.astype(np.complex128)
            ptr = nfftlib.jnfft_set_fhat(self.plan, fhat_complex.ctypes.data_as(ctypes.POINTER(np.complex128)))
            self.fhat = ptr

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
                setattr(self, v, val)

    def getproperty(self, v):
        if v == 'x':
            if self.x is None:
                raise AttributeError("x is not set.")
            ptr = self.x
            if self.D == 1:
                return np.ctypeslib.as_array(ptr, shape=(self.M,)) # Get nodes from C memory and convert to Python type
            else:
                return np.ctypeslib.as_array(ptr, shape=(self.D, self.M)) # Get nodes from C memory and convert to Python type
        
        elif v == 'num_threads':
            return nfftlib.nfft_get_num_threads()
        
        elif v == 'f':
            if self.f is None:
                raise AttributeError("f is not set.")
            ptr = self.f
            return np.ctypeslib.as_array(ptr, shape=(self.M,)) # Get function values from C memory and convert to Python type
        
        elif v == 'fhat':
            if self.fhat is None:
                raise AttributeError("fhat is not set.")
            ptr = self.fhat
            return np.ctypeslib.as_array(ptr, shape=(np.prod(self.N),)) # Get function values from C memory and convert to Python type
        
        else:
            return self.__dict__[v]

    # # nfft trafo direct [call with NFFT.trafo_direct outside module]
    # """
    #     nfft_trafo_direct(P)

    # computes the NDFT via naive matrix-vector multiplication for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``\hat{f}_{\pmb{k}} \in \mathbb{C}, \pmb{k} \in I_{\pmb{N}}^D,`` in `P.fhat`.

    # # Input
    # * `P` - a NFFT plan structure.

    # # See also
    # [`NFFT{D}`](@ref), [`nfft_trafo`](@ref)
    # """

    def nfft_trafo_direct(self):
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if self.fhat is None:
            raise ValueError("fhat has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")

        ptr = nfftlib.jnfft_trafo_direct(self.plan)
        self.f = ptr

    def trafo_direct(self):
        return self.nfft_trafo_direct()

    # # adjoint trafo direct [call with NFFT.adjoint_direct outside module]
    # """
    #     nfft_adjoint_direct(P)

    # computes the adjoint NDFT via naive matrix-vector multiplication for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``f_j \in \mathbb{C}, j =1,2,\dots,M,`` in `P.f`.

    # # Input
    # * `P` - a NFFT plan structure.

    # # See also
    # [`NFFT{D}`](@ref), [`nfft_adjoint`](@ref)
    # """

    def nfft_adjoint_direct(self):
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if not hasattr(self, 'f'):
            raise ValueError("f has not been set.")

        if not hasattr(self, 'x'):
            raise ValueError("x has not been set.")

        ptr = nfftlib.jnfft_adjoint_direct(self.plan)
        self.fhat = ptr

    def adjoint_direct(self):
        return self.nfft_adjoint_direct()

    # # nfft trafo [call with NFFT.trafo outside module]
    # """
    #     nfft_trafo(P)

    # computes the NDFT via the fast NFFT algorithm for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``\hat{f}_{\pmb{k}} \in \mathbb{C}, \pmb{k} \in I_{\pmb{N}}^D,`` in `P.fhat`.

    # # Input
    # * `P` - a NFFT plan structure.

    # # See also
    # [`NFFT{D}`](@ref), [`nfft_trafo_direct`](@ref)
    # """

    def nfft_trafo(self):
        nfftlib.jnfft_trafo.argtypes = [ctypes.POINTER(nfft_plan)]
        nfftlib.jnfft_trafo.restype = np.ctypeslib.ndpointer(np.complex128, shape=(self.M,), flags='C')
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if not hasattr(self, 'fhat'):
            raise ValueError("fhat has not been set.")

        if not hasattr(self, 'x'):
            raise ValueError("x has not been set.")
        
        # attributes = [
        #     'N', 'M', 'n', 'm', 'D', 'f1', 'f2',
        #     'init_done', 'finalized', 'x', 'f', 'fhat', 'plan'
        # ]
        
        # for attr in attributes:
        #     value = getattr(self, attr)
        #     print(f"{attr} (type: {type(value)}): {value}")

        ptr = nfftlib.jnfft_trafo(self.plan)
        print("PTR=",ptr)

        self.f = ptr

    def trafo(self):
        return self.nfft_trafo()

    # # adjoint trafo [call with NFFT.adjoint outside module]
    # """
    #     nfft_adjoint(P)

    # computes the adjoint NDFT via the fast adjoint NFFT algorithm for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``f_j \in \mathbb{C}, j =1,2,\dots,M,`` in `P.f`.

    # # Input
    # * `P` - a NFFT plan structure.

    # # See also
    # [`NFFT{D}`](@ref), [`nfft_adjoint_direct`](@ref)
    # """

    def nfft_adjoint(self):
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFFT already finalized")

        if not hasattr(self, 'f'):
            raise ValueError("f has not been set.")

        if not hasattr(self, 'x'):
            raise ValueError("x has not been set.")

        ptr = nfftlib.jnfft_adjoint(self.plan)
        self.fhat = ptr

    def adjoint(self):
        return self.nfft_adjoint()