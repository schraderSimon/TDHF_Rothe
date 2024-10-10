from sympy import *
import sympy as sp
a=sp.symbols('a',real=True,positive=True)
x, b, p, q, creal, cimag = sp.symbols('x b p q creal cimag',real=True)
def sympy_to_python(expr):
    return sp.lambdify((x, a, b, p, q), expr, modules=['numpy'])
def sympy_to_python_xonly(expr):
    return sp.lambdify(x, expr, modules=['numpy'])

gauss_expr=sqrt(sqrt(a**2)/sqrt(pi/2))*sp.exp(-(a**2+1j*b)*(x-q)**2+ 1j*p*(x-q))

minus_half_laplace_expr=sp.simplify(-0.5*sp.diff(gauss_expr,x,2))

gauss_expr_derivs=[]
minus_half_laplace_expr_derivs=[]
deriv_vals=[a,b,p,q]
variables=["a","b","p","q"]
for deriv_val in deriv_vals:
    gauss_expr_derivs.append(simplify(sp.diff(gauss_expr, deriv_val, 1)))
    minus_half_laplace_expr_derivs.append((sp.diff(minus_half_laplace_expr, deriv_val, 1)))
for i,variable in enumerate(variables):
    print("Derivative of gauss_expr with respect to %s:"%variable)
    print(simplify(gauss_expr_derivs[i]/gauss_expr))
for i,variable in enumerate(variables):
    print("Derivative of minus_half_laplace_expr with respect to %s:"%variable)
    print(simplify(minus_half_laplace_expr_derivs[i]/minus_half_laplace_expr))