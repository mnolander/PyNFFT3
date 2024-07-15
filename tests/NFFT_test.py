import warnings
import ctypes
import numpy as np
from src.pynfft3.flags import *
from src.pynfft3.NFFT import *

# import numpy as np
N = np.array([16], dtype='int32') # 1d
# N = np.array([16, 8], dtype='int32') # 2d
# N = np.array([16, 8, 4], dtype='int32') # 3d
M = 100
d = len(N)
Ns = np.prod(N)

X = np.array([[np.sin(i + j) for j in range(d)] for i in range(M)])
fhat = np.array([np.cos(k) + 1.0j * np.sin(k) for k in range(Ns)])
f = np.array([np.sin(m) + 1.0j * np.cos(m) for m in range(M)])

# test init and setting
plan = NFFT(N,M)
plan.X = X
plan.f = f   # this gets overwritten
plan.fhat = fhat

# test traffo
plan.trafo() # value is in plan.f

# compare with directly computed
I = [[k] for  k in range(int(-N[0]/2),int(N[0]/2))] # 1d
# I = [[k, i] for  k in range(int(-N[0]/2),int(N[0]/2)) for i in range(int(-N[1]/2),int(N[1]/2))] # 2d
# I = [[k, i, j] for  k in range(int(-N[0]/2),int(N[0]/2)) for i in range(int(-N[1]/2),int(N[1]/2)) for j in range(int(-N[2]/2),int(N[2]/2))] # 3d

F = np.array([[np.exp(-2 * np.pi * 1j * np.dot(X.T[:,j],I[l])) for l in range (0,Ns) ] for j in range(0,M)])

f1 = F @ fhat

norm_euclidean = np.linalg.norm(f1 - plan.f)
norm_infinity = np.linalg.norm(f1 - plan.f, np.inf)

print("Euclidean norm:", norm_euclidean)
print("Infinity norm:", norm_infinity)