import warnings
import ctypes
import numpy as np
from src.pynfft3.flags import *
from src.pynfft3.NFFT import *

n = np.array([6])
N = np.array([2]) 
M = 2

test_nfft = create_NFFT(N, M)
test_nfft_default = create_NFFT_with_defaults(N, M, n)
print(test_nfft)
print(test_nfft_default)

nfft_init(test_nfft)