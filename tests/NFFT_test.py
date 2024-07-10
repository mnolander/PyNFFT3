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

fhat_val = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex128)
setproperty(test_nfft, "fhat", fhat_val)