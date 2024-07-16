import warnings
import ctypes
import numpy as np
import atexit
from .flags import *
from . import nfstlib
from . import nfst_plan

# # NFST plan struct
# """
#     NFST{D}

# A NFST (nonequispaced fast sine transform) plan, where D is the dimension. 

# The NFST realizes a direct and fast computation of the discrete nonequispaced sine transform. The aim is to compute

# ```math
# f^s(\pmb{x}_j) = \sum_{\pmb{k} \in I_{\pmb{N},\mathrm{s}}^D} \hat{f}_{\pmb{k}}^s \, \sin(2\pi \, \pmb{k} \odot \pmb{x}_j)
# ```

# at given arbitrary knots ``\pmb{x}_j \in [0,0.5]^D, j = 1, \cdots, M``, for coefficients ``\hat{f}^{s}_{\pmb{k}} \in \mathbb{R}``, ``\pmb{k} \in I_{\pmb{N},\mathrm{s}}^D \coloneqq \left\{ \pmb{k} \in \mathbb{Z}^D: 1 \leq k_i \leq N_i - 1, \, i = 1,2,\ldots,D \right\}``, and a multibandlimit vector ``\pmb{N} \in \mathbb{N}^{D}``. Note that we define ``\sin(\pmb{k} \circ \pmb{x}) \coloneqq \prod_{i=1}^D \sin(k_i \cdot x_i)``. The transposed problem reads as

# ```math
# \hat{h}^s_{\pmb{k}} = \sum_{j=1}^M f^s_j \, \sin(2\pi \, \pmb{k} \odot \pmb{x}_j)
# ```

# for the frequencies ``\pmb{k} \in I_{\pmb{N},\mathrm{s}}^D`` with given coefficients ``f^s_j \in \mathbb{R}, j = 1,2,\ldots,M``.

# # Fields
# * `N` - the multibandlimit ``(N_1, N_2, \ldots, N_D)`` of the trigonometric polynomial ``f^s``.
# * `M` - the number of nodes.
# * `n` - the oversampling ``(n_1, n_2, \ldots, n_D)`` per dimension.
# * `m` - the window size. A larger m results in more accuracy but also a higher computational cost. 
# * `f1` - the NFST flags.
# * `f2` - the FFTW flags.
# * `init_done` - indicates if the plan is initialized.
# * `finalized` - indicates if the plan is finalized.
# * `x` - the nodes ``x_j \in [0,0.5]^D, \, j = 1, \ldots, M``.
# * `f` - the values ``f^s(\pmb{x}_j)`` for the NFST or the coefficients ``f_j^s \in \mathbb{R}, j = 1, \ldots, M,`` for the transposed NFST.
# * `fhat` - the Fourier coefficients ``\hat{f}_{\pmb{k}}^s \in \mathbb{R}`` for the NFST or the values ``\hat{h}_{\pmb{k}}^s, \pmb{k} \in I_{\pmb{N},\mathrm{s}}^D,`` for the adjoint NFFT.
# * `plan` - plan (C pointer).

# # Constructor
#     NFST{D}( N::NTuple{D,Int32}, M::Int32, n::NTuple{D,Int32}, m::Int32, f1::UInt32, f2::UInt32 ) where {D}

# # Additional Constructor
#     NFST( N::NTuple{D,Int32}, M::Int32, n::NTuple{D,Int32}, m::Int32, f1::UInt32, f2::UInt32) where {D}
#     NFST( N::NTuple{D,Int32}, M::Int32) where {D}

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

nfstlib.jfnst_alloc.restype = ctypes.POINTER(nfst_plan)
nfstlib.jfnst_finalize.argtypes = [ctypes.POINTER(nfst_plan)]

nfstlib.jfnst_set_x.argtypes = [ctypes.POINTER(nfst_plan), np.ctypeslib.ndpointer(np.float64, flags='C')]
nfstlib.jfnst_set_f.argtypes = [ctypes.POINTER(nfst_plan), np.ctypeslib.ndpointer(np.float128, ndim=1, flags='C')] 
nfstlib.jfnst_set_fhat.argtypes = [ctypes.POINTER(nfst_plan), np.ctypeslib.ndpointer(np.float128, ndim=1, flags='C')] 

nfstlib.jfnst_trafo.argtypes = [ctypes.POINTER(nfst_plan)]
nfstlib.jfnst_adjoint.argtypes = [ctypes.POINTER(nfst_plan)]
nfstlib.jfnst_trafo_direct.argtypes = [ctypes.POINTER(nfst_plan)]
nfstlib.jfnst_adjoint_direct.argtypes = [ctypes.POINTER(nfst_plan)]

class NFST:
    def __init__(self, N, M, n=None, m=default_window_cut_off, f1=None, f2=f2_default):
        self.plan = nfstlib.jnfst_alloc()
        self.N = N  # bandwidth tuple
        N_ct = N.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self.M = M  # number of nodes
        self.n = n  # oversampling per dimension
        self.m = m  # window size
        self.D = len(N)  # dimensions

        if any(x <= 0 for x in N):
            raise ValueError(f"Invalid N: {N}. Argument must be a positive integer")
        
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
        nfstlib.jnfst_init(self.plan, self.D, N_ct, self.M, n_ct, self.m, self.f1, self.f2)
        self.init_done = True  # bool for plan init
        self.finalized = False  # bool for finalizer
        self.x = None  # nodes, will be set later
        self.f = None  # function values
        self.fhat = None  # Fourier coefficients
    
    # # finalizer
    # """
    #     nfst_finalize_plan(P)

    # destroys a NFST plan structure.

    # # Input
    # * `P` - a NFST plan structure.

    # # See also
    # [`NFST{D}`](@ref), [`nfst_init`](@ref)
    # """

    def nfst_finalize_plan(self):
        nfstlib.jnfst_finalize.argtypes = (
            ctypes.POINTER(nfst_plan),   # P
        )

        if not self.init_done:
            raise ValueError("NFST not initialized.")

        if not self.finalized:
            self.finalized = True
            nfstlib.jnfst_finalize(self.plan)

    def finalize_plan(self):
        return self.nfst_finalize_plan()

    # # allocate plan memory and init with D,N,M,n,m,f1,f2
    # """
    #     nfst_init(P)

    # intialises the NFST plan in C. This function does not have to be called by the user.

    # # Input
    # * `p` - a NFST plan structure.

    # # See also
    # [`NFST{D}`](@ref), [`nfst_finalize_plan`](@ref)
    # """

    def nfst_init(self):
        # Convert N and n to numpy arrays for passing them to C
        Nv = np.array(self.N, dtype=np.int32)
        n = np.array(self.n, dtype=np.int32)

        # Call init for memory allocation
        ptr = nfstlib.jnfst_alloc()

        # Set the pointer
        self.plan = ctypes.cast(ptr, ctypes.POINTER(nfst_plan))

        # Initialize values
        nfstlib.jnfst_init(
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

        atexit.register(self.nfst_finalize_plan())

    def init(self):
        return self.nfst_init()
    
    @property
    def X(self):
        return self._X

    @X.setter 
    def X(self, value):
        if value is not None:
            if not (isinstance(value,np.ndarray) and value.dtype == np.float64 and value.flags['C']):
                raise RuntimeError("X has to be C-continuous, numpy float64 array")
            if self.D == 1:
                nfstlib.jnfst_set_x.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=self.M, flags='C')
            else:
                nfstlib.jnfst_set_x.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=(self.M,self.D), flags='C')
            self._X = nfstlib.jnfst_set_x(self.plan, value)
    
    @property
    def f(self):
        return self._f
    
    @f.setter 
    def f(self, value):
        if value is not None:
            if not (isinstance(value,np.ndarray) and value.dtype == np.float128 and value.flags['C']):
                raise RuntimeError("f has to be C-continuous, numpy float128 array")
            nfstlib.jnfst_set_f.restype = np.ctypeslib.ndpointer(np.float128, ndim=1, shape=self.M, flags='C') 
            self._f = nfstlib.jnfst_set_f(self.plan, value)

    @property
    def fhat(self):
        return self._fhat
    
    @fhat.setter 
    def fhat(self, value):
        Ns = np.prod(self.N - 1)
        if value is not None:
            if not (isinstance(value,np.ndarray) and value.dtype == np.float128 and value.flags['C']):
                raise RuntimeError("fhat has to be C-continuous, numpy float128 array") 
            if value.size != Ns:
                raise ValueError(f"fhat has to be an array of size {Ns}")
            nfstlib.jnfst_set_fhat.restype = np.ctypeslib.ndpointer(np.float128, ndim=1, shape=Ns, flags='C') 
            self._fhat = nfstlib.jnfst_set_fhat(self.plan, value)

    # """
    #     nfst_trafo_direct(P)

    # computes the NDST via naive matrix-vector multiplication for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``\hat{f}_{\pmb{k}}^s \in \mathbb{R}, \pmb{k} \in I_{\pmb{N},s}^D,`` in `P.fhat`.

    # # Input
    # * `P` - a NFST plan structure.

    # # See also
    # [`NFST{
    # """

    def nfst_trafo_direct(self):
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFST already finalized")

        if self.fhat is None:
            raise ValueError("fhat has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")

        ptr = nfstlib.jnfst_trafo_direct(self.plan)
        self.f = ptr

    def trafo_direct(self):
        return self.nfst_trafo_direct()

    # """
    #     nfst_transposed_direct(P)

    # computes the transposed NDST via naive matrix-vector multiplication for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``f_j^s \in \mathbb{R}, j =1,2,\dots,M,`` in `P.f`.

    # # Input
    # * `P` - a NFST plan structure.

    # # See also
    # [`NFST{D}`](@ref), [`nfst_transposed`](@ref)
    # """

    def nfst_transposed_direct(self):
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFST already finalized")

        if self.f is None:
            raise ValueError("f has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")
        
        ptr = nfstlib.jnfst_adjoint_direct(self.plan)
        self.fhat = ptr

    def nfst_adjoint_direct(self):
        return self.nfst_transposed_direct()

    def adjoint_direct(self):
        return self.nfst_adjoint_direct()

    # # nfft trafo [call with NFFT.trafo outside module]
    # """
    #     nfft_trafo(P)

    # computes the NDFT via the fast NFFT algorithm for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``\hat{f}_{\pmb{k}} \in \mathbb{C}, \pmb{k} \in I_{\pmb{N}}^D,`` in `P.fhat`.

    # # Input
    # * `P` - a NFFT plan structure.

    # # See also
    # [`NFFT{D}`](@ref), [`nfft_trafo_direct`](@ref)
    # """

    def nfst_trafo(self):
        nfstlib.jnfst_trafo.restype = np.ctypeslib.ndpointer(np.float128, shape=self.M, flags='C')
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFST already finalized")

        if not hasattr(self, 'fhat'):
            raise ValueError("fhat has not been set.")

        if not hasattr(self, 'x'):
            raise ValueError("x has not been set.")

        ptr = nfstlib.jnfst_trafo(self.plan)
        self.f = ptr

    def trafo(self):
        return self.nfst_trafo()

    # """
    #     nfst_transposed(P)

    # computes the transposed NDST via the fast transposed NFST algorithm for provided nodes ``\pmb{x}_j, j =1,2,\dots,M,`` in `P.X` and coefficients ``f_j^s \in \mathbb{R}, j =1,2,\dots,M,`` in `P.f`.

    # # Input
    # * `P` - a NFST plan structure.

    # # See also
    # [`NFST{D}`](@ref), [`nfst_transposed_direct`](@ref)
    # """

    def nfst_transposed(self):
        # Prevent bad stuff from happening
        if self.finalized:
            raise RuntimeError("NFST already finalized")

        if self.f is None:
            raise ValueError("f has not been set.")

        if self.x is None:
            raise ValueError("x has not been set.")
        
        ptr = nfstlib.jnfst_adjoint(self.plan)
        self.fhat = ptr

    def nfst_adjoint(self):
        return self.nfst_transposed()

    def adjoint(self):
        return self.nfst_adjoint()