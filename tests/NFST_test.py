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
plan.trafo() # value is in plan.f

# test transpose
# plan.adjoint()

# compare with directly computed, using N[k]-1 but range is not inclusive
if d == 1:
    I = [[k] for k in range(1, N[0])]
elif d == 2:
    I = [[k, i] for k in range(1, N[0]) for i in range(1, N[1])]
elif d == 3:
    I = [[k, i, j] for k in range(1, N[0]) for i in range(1, N[1]) for j in range(1, N[2])]

def sine_product(x, i, d):
    result = 1
    for k in range(d):
        result *= np.sin(2 * np.pi * x[k] * i[k])
    return result

F = np.array([[sine_product(X[j], I[l], d) for l in range(Ns)] for j in range(M)])
F_mat = np.asmatrix(F)

# for testing trafo
f1 = F @ fhat
norm_euclidean = np.linalg.norm(f1 - plan.f)
norm_infinity = np.linalg.norm(f1 - plan.f, np.inf)

# # for testing transpose
# f1 = F_mat.H @ f
# norm_euclidean = np.linalg.norm(f1 - plan.fhat)
# norm_infinity = np.linalg.norm(f1 - plan.fhat, np.inf)

print("Euclidean norm:", norm_euclidean)
print("Infinity norm:", norm_infinity)