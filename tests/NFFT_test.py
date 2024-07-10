import warnings
import ctypes
import numpy as np
from src.pynfft3.flags import *
from src.pynfft3.NFFT import *

# Set inital values
n = np.array([6])
N = np.array([2]) 
M = 2

# Test constructor
test_nfft = create_NFFT(N, M)
#test_nfft = create_NFFT_with_defaults(N, M, n)
print(test_nfft)

# Test init
#init(test_nfft)
nfft_init(test_nfft)

# Test setting x, fhat, and f
x_val = np.array([3.0, 4.0], dtype=np.float64)
f_val = np.array([1+1j, 2+2j], dtype=np.complex128)
fhat_val = np.array([6+7j, 2+3j], dtype=np.complex128)
setproperty(test_nfft, "x", x_val)
setproperty(test_nfft, "fhat", fhat_val)
setproperty(test_nfft, "f", f_val)

# Test retrieving x, fhat, and f
print("P.x:",getproperty(test_nfft, "x"))
print("P.fhat:",getproperty(test_nfft, "fhat"))
print("P.f:",getproperty(test_nfft, "f"))

nfft_trafo_direct(test_nfft)
#trafo_direct(test_nfft)
print("P.f:",getproperty(test_nfft, "f"))

nfft_adjoint_direct(test_nfft)
#adjoint_direct(test_nfft)
print("P.fhat:",getproperty(test_nfft, "fhat"))

nfft_trafo(test_nfft)
#trafo(test_nfft)
print("P.f:",getproperty(test_nfft, "f"))

nfft_adjoint(test_nfft)
#adjoint(test_nfft)
print("P.fhat:",getproperty(test_nfft, "fhat"))