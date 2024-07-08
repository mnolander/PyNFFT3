import ctypes
import numpy as np
from numpy import ctypeslib
import os

class nfftplan(ctypes.Structure):
    pass

class ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

nfftlib = ctypes.CDLL(os.path.dirname(__file__)+"\\libnfftjulia.dll")