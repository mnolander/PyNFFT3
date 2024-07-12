import warnings
import ctypes
import numpy as np
from src.pynfft3.flags import *
from src.pynfft3.NFFT import *

# # Set inital values
# n = np.array([6])
# N = np.array([2]) 
# M = 2

# import numpy as np
N = np.array([16, 8, 4], dtype='int32') # pretty important, should add a check for that dtype, lost 3h here
M = 100
d = len(N)
Ns = np.prod(N)

X = np.random.rand(M,d)
fhat = np.random.rand(Ns) +  1.0j * np.random.rand(Ns)
f = np.random.rand(M) +  1.0j * np.random.rand(M)

# Test constructor
plan = NFFT(N, M)
#plan = create_NFFT_with_defaults(N, M, n)
print(plan)

print(plan.f1)
print(plan.f2)

# Test init
#init(plan)
nfft_init(plan)

# Test setting x, fhat, and f
x_val = np.array([3.0, 4.0], dtype=np.float64)
f_val = np.array([1+1j, 2+2j], dtype=np.complex128)
fhat_val = np.array([6+7j, 2+3j], dtype=np.complex128)
setproperty(plan, "x", x_val)
setproperty(plan, "fhat", fhat_val)
setproperty(plan, "f", f_val)

# Test retrieving x, fhat, and f
print("P.x:",getproperty(plan, "x"))
print("P.fhat:",getproperty(plan, "fhat"))
print("P.f:",getproperty(plan, "f"))

nfft_trafo_direct(plan)
#trafo_direct(plan)
print("P.f:",getproperty(plan, "f"))

nfft_adjoint_direct(plan)
#adjoint_direct(plan)
print("P.fhat:",getproperty(plan, "fhat"))

nfft_trafo(plan)
#trafo(plan)
print("P.f:",getproperty(plan, "f"))

nfft_adjoint(plan)
#adjoint(plan)
print("P.fhat:",getproperty(plan, "fhat"))