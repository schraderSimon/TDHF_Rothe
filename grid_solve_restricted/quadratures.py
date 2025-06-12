import numpy as np
from scipy.special import roots_legendre

def gaussian_quadrature(a, b, n):
    x, w = roots_legendre(n)
    points = 0.5 * (b - a) * x + 0.5 * (b + a)
    weights = 0.5 * (b - a) * w   
    return points, weights
def trapezoidal_quadrature(a, b, n):
    points = np.linspace(a, b, n)
    
    dx = points[1] - points[0]
    
    weights = np.full(n, dx)
    weights[0] *= 0.5  
    weights[-1] *= 0.5 
    
    return points, weights
