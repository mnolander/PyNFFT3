import warnings
import ctypes
import numpy as np
from src.pynfft3.flags import *
from src.pynfft3.fastsum import *

d = 2
N = 5
M = 10
kernel = "multiquadric"
c = 1 / np.sqrt(N)
eps_B = 1 / 16

# Create a FASTSUM object in Python
plan = FASTSUM(d, N, M, kernel, c)

print(f"FASTSUM: (d={plan.d}, N={plan.N}, M={plan.M}, n={plan.n}, p={plan.p}, kernel={plan.kernel}, "
            f"c={plan.c}, eps_I={plan.eps_I}, eps_B={plan.eps_B}, nn_x={plan.nn_x}, nn_y={plan.nn_y}, "
            f"m_x={plan.m_x}, m_y={plan.m_y}, init_done={plan.init_done}, finalized={plan.finalized}, flags={plan.flags})")

# Generate source nodes in circle of radius 0.25 - eps_B / 2
r = np.sqrt(np.random.rand(N)) * (0.25 - eps_B / 2)
phi = np.random.rand(N) * (2 * np.pi)
X = np.column_stack((r * np.cos(phi), r * np.sin(phi)))
print("X:",X)
plan.x = X

# Generate coefficients alpha_k
alpha = np.random.rand(N) + 1j * np.random.rand(N)
plan.alpha = alpha

# Generate target nodes in circle of radius 0.25 - eps_B / 2
r = np.sqrt(np.random.rand(M)) * (0.25 - eps_B / 2)
phi = np.random.rand(M) * (2 * np.pi)
Y = np.column_stack((r * np.cos(phi), r * np.sin(phi)))
plan.y = Y