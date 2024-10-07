import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import numba
from numba import prange,jit
import sys
import sympy as sp
from scipy.special import roots_legendre
from grid_HF import Molecule1D, Coulomb
from scipy import linalg
import time
from numpy.polynomial.hermite import hermgauss

from scipy.optimize import minimize
from numpy import array,sqrt
from numpy import exp
from numpy import cosh, tanh, arctanh


def trapezoidal_quadrature(a, b, n):
    # Generate equally spaced points in the interval [a, b]
    points = np.linspace(a, b, n)
    
    # Calculate the spacing between points
    dx = points[1] - points[0]
    
    # Calculate weights for the trapezoidal rule
    weights = np.full(n, dx)
    weights[0] *= 0.5  # Half weight for the first point
    weights[-1] *= 0.5  # Half weight for the last point
    
    return points, weights

