import warnings
import ctypes
import numpy as np
from src.pynfft3.flags import *
from src.pynfft3.NFFT import *

n = np.array([6])
N = np.array([2]) 
M = 2

test_nfft = create_NFFT(N, M)
#test_nfft = create_NFFT_with_defaults(N, M, n)
print(test_nfft)

#init(test_nfft)
nfft_init(test_nfft)

x_val = np.array([3.0, 4.0], dtype=np.float64)
f_val = np.array([1+1j, 2+2j], dtype=np.complex128)
fhat_val = np.array([6+7j, 2+3j], dtype=np.complex128)
setproperty(test_nfft, "x", x_val)
setproperty(test_nfft, "fhat", fhat_val)
setproperty(test_nfft, "f", f_val)
print("P.x:", test_nfft.x)
print("P.fhat:", test_nfft.fhat)
print("P.f:", test_nfft.f)