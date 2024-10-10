import numpy as np
from numpy import pi, sqrt, exp
# Constants
h = 1e-4  # Step size for numerical differentiation

# Function to calculate Gaussian, minus_half_laplacian, and derivatives
def gauss_and_minushalflaplacian_and_derivs(x,a,b,p,q):
    asq=a**2
    bredde = asq + 1j*b
    qminx = q - x
    jp=1j*p
    br_qminx=bredde*qminx
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + br_qminx))
    minus_half_laplace = (bredde - 0.5 * (jp + 2*br_qminx)**2) * gaussval
    
    aderiv=(-2*a*q**2 + 4*a*q*x - 2*a*x**2 + 1/(2*a))*gaussval
    bderiv=1j*(-q**2 + 2.0*q*x - x**2)*gaussval
    pderiv=1j*(-q + x)*gaussval
    qderiv=(-2.0*asq*q + 2.0*asq*x - 2j*b*q + 2j*b*x - jp)*gaussval
    aderiv_kin=minus_half_laplace*(-asq*(qminx**2*(4*asq + 4*1j*b - 8.0*(0.5j*p + br_qminx)**2) + 16*qminx*(0.5*jp + br_qminx) - 4) 
                                   + bredde - 2*(0.5*jp + br_qminx)**2)/(2*a*(bredde - 2*(0.5*jp + br_qminx)**2))
    bderiv_kin=minus_half_laplace*1j*(-qminx**2*(bredde - 2.0*(0.5*jp + br_qminx)**2) - 
                                      4*qminx*(0.5*jp + br_qminx) + 1)/(bredde - 2.0*(0.5*jp + br_qminx)**2)
    pderiv_kin=minus_half_laplace*1j*(-jp - 2.0*br_qminx - qminx*(bredde - 2.0*(0.5*jp + br_qminx)**2))/(bredde - 2.0*(0.5*jp + br_qminx)**2)
    qderiv_kin=minus_half_laplace*(-2.0*(2*asq + 2j*b)*(0.5*jp + br_qminx) 
                                   - (jp + 2*br_qminx)*(bredde - 2.0*(0.5*jp + br_qminx)**2))/(bredde - 2.0*(0.5*jp + br_qminx)**2)
    return (gaussval, minus_half_laplace, aderiv, bderiv, pderiv, qderiv, 
            aderiv_kin, bderiv_kin, pderiv_kin, qderiv_kin)

# Central difference approximation
def central_difference(f, param_index, x, a, b, p, q):
    params = [x, a, b, p, q]
    params[param_index] += h
    f_plus_h = f(*params)
    
    params[param_index] -= 2 * h
    f_minus_h = f(*params)
    
    return (f_plus_h - f_minus_h) / (2 * h)

# Numerical derivative verification function
def verify_derivatives(x, a, b, p, q):
    # Analytical results
    _, _, aderiv, bderiv, pderiv, qderiv, aderiv_kin, bderiv_kin, pderiv_kin, qderiv_kin = gauss_and_minushalflaplacian_and_derivs(x, a, b, p, q)
    
    # Numerical differentiation
    gauss_f = lambda x, a, b, p, q: gauss_and_minushalflaplacian_and_derivs(x, a, b, p, q)[0]
    laplacian_f = lambda x, a, b, p, q: gauss_and_minushalflaplacian_and_derivs(x, a, b, p, q)[1]

    # Numerical derivatives
    aderiv_num = central_difference(gauss_f, 1, x, a, b, p, q)
    bderiv_num = central_difference(gauss_f, 2, x, a, b, p, q)
    pderiv_num = central_difference(gauss_f, 3, x, a, b, p, q)
    qderiv_num = central_difference(gauss_f, 4, x, a, b, p, q)

    aderiv_kin_num = central_difference(laplacian_f, 1, x, a, b, p, q)
    bderiv_kin_num = central_difference(laplacian_f, 2, x, a, b, p, q)
    pderiv_kin_num = central_difference(laplacian_f, 3, x, a, b, p, q)
    qderiv_kin_num = central_difference(laplacian_f, 4, x, a, b, p, q)

    # Compare numerical and analytical derivatives
    print("Analytical a derivative:", aderiv, "Numerical a derivative:", aderiv_num)
    print("Analytical b derivative:", bderiv, "Numerical b derivative:", bderiv_num)
    print("Analytical p derivative:", pderiv, "Numerical p derivative:", pderiv_num)
    print("Analytical q derivative:", qderiv, "Numerical q derivative:", qderiv_num)
    assert np.isclose(aderiv, aderiv_num, rtol=1e-4, atol=1e-4)
    assert np.isclose(bderiv, bderiv_num, rtol=1e-4, atol=1e-4)
    assert np.isclose(pderiv, pderiv_num, rtol=1e-4, atol=1e-4)
    assert np.isclose(qderiv, qderiv_num, rtol=1e-4, atol=1e-4)
    print("Analytical a kinetic derivative:", aderiv_kin, "Numerical a kinetic derivative:", aderiv_kin_num)
    print("Analytical b kinetic derivative:", bderiv_kin, "Numerical b kinetic derivative:", bderiv_kin_num)
    print("Analytical p kinetic derivative:", pderiv_kin, "Numerical p kinetic derivative:", pderiv_kin_num)
    print("Analytical q kinetic derivative:", qderiv_kin, "Numerical q kinetic derivative:", qderiv_kin_num)
    assert np.isclose(aderiv_kin, aderiv_kin_num, rtol=1e-4, atol=1e-4)
    assert np.isclose(bderiv_kin, bderiv_kin_num, rtol=1e-4, atol=1e-4)
    assert np.isclose(pderiv_kin, pderiv_kin_num, rtol=1e-4, atol=1e-4)
    assert np.isclose(qderiv_kin, qderiv_kin_num, rtol=1e-4, atol=1e-4)
# Example usage
x, a, b, p, q = 1, 2, 0.3, 0.7, 0.4
verify_derivatives(x, a, b, p, q)
