import warnings
import ctypes
import numpy as np
from src.pynfft3.flags import *
from src.pynfft3.NFST import *

# N = np.array([16], dtype='int32') # 1d
# N = np.array([16, 8], dtype='int32') # 2d
N = np.array([16, 8, 4], dtype='int32') # 3d

M = 100
d = len(N)
Ns = np.prod(N - 1)

X = np.array([[abs(np.sin(i + j)) for j in range(d)] for i in range(M)])
fhat = np.array([np.cos(k) * np.sin(k) for k in range(Ns)], dtype=np.float64)
f = np.array([np.sin(m) * np.cos(m) for m in range(M)])

# test init and setting
plan = NFST(N,M)
plan.x = X
plan.f = f # this gets overwritten
plan.fhat = fhat

# test traffo
# plan.trafo() # value is in plan.f

# test adjoint
plan.adjoint()

# compare with directly computed
# I = [[k] for  k in range(1,N[0]-1)] # 1d
# I = [[k, i] for  k in range(1,N[0]-1) for i in range(1,N[1]-1)] # 2d
I = [[k, i, j] for  k in range(1,N[0]-1) for i in range(1,N[1]-1) for j in range(1,N[2]-1)] # 3d

#F = np.array([[np.exp(-2 * np.pi * 1j * np.dot(X.T[:,j],I[l])) for l in range (0,Ns) ] for j in range(0,M)])
F = np.array([[np.sin(2*np.pi*X[1,j]*I[l][1]) * np.sin(2*np.pi*X[2,j]*I[l][2]) * np.sin(2*np.pi*X[3,j]*I[l][3]) for l in range (0,Ns) ] for j in range(0,M)])
F_mat = np.asmatrix(F)

# # for testing trafo
# f1 = F @ fhat
# norm_euclidean = np.linalg.norm(f1 - plan.fhat)
# norm_infinity = np.linalg.norm(f1 - plan.fhat, np.inf)

# for testing adjoint
f1 = F_mat.H @ f
norm_euclidean = np.linalg.norm(f1 - plan.fhat)
norm_infinity = np.linalg.norm(f1 - plan.fhat, np.inf)

print("Euclidean norm:", norm_euclidean)
print("Infinity norm:", norm_infinity)