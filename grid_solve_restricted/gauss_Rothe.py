import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numba
from numba import prange,jit
import sys
import os
import sympy as sp
from grid_HF import Molecule1D, Coulomb
from scipy import linalg
import time
from numpy.polynomial.hermite import hermgauss
import autograd.numpy as np
from autograd import grad, jacobian

from scipy.optimize import minimize
from numpy import array,sqrt, pi
from numpy import exp
from numpy import cosh, tanh, arctanh
#from sympy import *
from quadratures import *
np.set_printoptions(linewidth=300, precision=2, suppress=True, formatter={'float': '{:0.2e}'.format})
class ConvergedError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# Function to save or append data
def save_wave_function(filename, wave_function, time_step,xval,time):
    try:
        # Try to load the existing data
        existing_data = np.load(filename)
        params=existing_data['params']
        params=list(params)
        xvals=list(existing_data['xvals'])
        times=list(existing_data['times'])

    except FileNotFoundError:
        params=[]
        xvals=[]
        times=[]
    params.append(wave_function)
    xvals.append(xval)
    times.append(time)
    # Append new wave function at the current time step
    np.savez(filename, params=params, time_step=time_step,times=times,xvals=xvals)

def minimize_transformed_bonds(error_function,start_params,num_gauss,num_frozen,num_orbitals=2,multi_bonds=0.03,maxiter=20,gradient=None,both=False):
    """
    Minimizes with min_max bonds as described in https://lmfit.github.io/lmfit-py/bounds.html
    """
    def transform_params(untransformed_params):
        return arctanh(2*(untransformed_params-mins)/(maxs-mins)-1)
        #return arcsin(2*(untransformed_params-mins)/(maxs-mins)-1)
    def untransform_params(transformed_params):
        return mins+(maxs-mins)/2*(1+tanh(transformed_params))
        #return mins+(maxs-mins)/2*(1+sin(transformed_params))
    def chainrule_params(transformed_params):
        returnval= 0.5*(maxs-mins)/(cosh(transformed_params)**2)
        return returnval
        #return 0.5*(maxs-mins)*cos(transformed_params)

    def transformed_error(transformed_params):
        error=error_function(untransform_params(transformed_params),num_gauss)
        return error
    def transformed_gradient(transformed_params):
        orig_grad=gradient(untransform_params(transformed_params),num_gauss)
        chainrule_grad=chainrule_params(transformed_params)
        grad=orig_grad*chainrule_grad
        if(np.isnan(sum(grad))):

            print("gradient has nan")
            print(grad)
            print(transformed_error(transformed_params))
            if np.isnan(np.sum(orig_grad)):
                print("Original gradient has nan...")
            return np.nan_to_num(grad)
        return grad
    def transform_error_and_gradient(transformed_params):
        untransformed_params=untransform_params(transformed_params)
        error,orig_grad=error_function(untransformed_params,num_gauss)
        chainrule_grad=chainrule_params(transformed_params)
        grad=orig_grad*chainrule_grad
        if(np.isnan(sum(grad))):

            print("gradient has nan")
            print(grad)
            print(transformed_error(transformed_params))
            if np.isnan(np.sum(orig_grad)):
                print("Original gradient has nan...")
            return np.nan_to_num(grad)
        return error,grad

    dp=multi_bonds*np.ones(len(start_params)) #Percentage (times 100) how much the parameters are alowed to change compared to previous time step
    range_nonlin=[0.01,0.1,0.1,0.1]*(num_gauss-num_frozen)
    rangex=range_nonlin
    rangex=np.array(rangex)
    mins=start_params-rangex-dp*abs(start_params)
    maxs=start_params+rangex+dp*abs(start_params)
    mmins=np.zeros_like(mins)
    mmaxs=np.zeros_like(maxs)
    mmins[0::4]=0.05
    mmaxs[0::4]=10
    mmins[1::4]=-30
    mmaxs[1::4]=30
    mmins[2::4]=-50
    mmaxs[2::4]=50
    mmaxs[3::4]=grid_b-10
    mmins[3::4]=grid_a+10
    for i in range(len(mins)):
        if mins[i]<mmins[i]:
            mins[i]=mmins[i]
        if maxs[i]>mmaxs[i]:
            maxs[i]=mmaxs[i]
    transformed_params=transform_params(start_params)
    transformed_params=np.real(np.nan_to_num(transformed_params))

    start=time.time()
    if both:
        err0,grad0=transform_error_and_gradient(transformed_params)
    else:
        grad0=transformed_gradient(transformed_params)
    hess_inv0=np.diag(1/abs(grad0+1e-16*np.array(len(grad0))))
    if both is False:
        sol=minimize(transformed_error,transformed_params,jac=transformed_gradient,method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':3e-9})
    else:
        sol=minimize(transform_error_and_gradient,transformed_params,jac=True,method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':3e-9})
    
    transformed_sol=sol.x
    end=time.time()
    print("  REG: Time to optimize: %.3f seconds, niter : %d"%(end-start,sol.nit))
    return untransform_params(transformed_sol), sol.fun

n=1001
grid_a, grid_b = -60, 60
lambd=1e-8 #Should be at most 1e-8, otherwise the <x(t)> will become wrongly oscillating
#points, weights = tanh_sinh_quadrature(grid_a, grid_b, n)
#points, weights = trapezoidal_quadrature(grid_a, grid_b, n)
points,weights=gaussian_quadrature(grid_a, grid_b, n)

sqrt_weights=np.sqrt(weights)
R_list=[-1.15, 1.15]
Z_list=[3,1]
alpha=0.5
norbs=2
typer=sys.argv[3]

def calculate_potential(Z_list,R_list,alpha,points):
    V=Molecule1D(R_list,Z_list,alpha)(points)
    return V
def e_e_interaction(x):
    """
    Calculate the electron-electron interaction term
    """
    Vee = np.zeros((len(x), len(x)),dtype=np.complex128)
    for i in range(len(x)):
        for j in range(len(x)):
            Vee[i, j] = Coulomb(x[i] - x[j],x_c=0, Z=-1, alpha=1)
    return Vee
V=external_potential=calculate_potential(Z_list,R_list,alpha,points)


#Initial Parameters from fit_gaussians_to_initial_orbitals.py
params0=[np.float64(-1.7221769956476094), np.float64(0.14972641307166773), np.float64(-0.005214530627800701), np.float64(-1.1626332649725455), np.float64(0.08938953120866812), np.float64(-0.012817948096138197), np.float64(-1.0095819437514282), np.float64(0.10028234383283993), np.float64(-0.003912463144352765), np.float64(-1.1396915104863552), np.float64(0.4323715059907284), np.float64(-0.08251341567662228), np.float64(-0.5824064153267005), np.float64(0.13177148588224852), np.float64(0.7413271149275448), np.float64(-0.21038791495968723), np.float64(-0.13701292603493387), np.float64(0.0064189173825366565), np.float64(-0.49383671013656044), np.float64(-0.02806401161803651), np.float64(0.06716925708896938), np.float64(-0.6793977807901034), np.float64(0.3071508961137962), np.float64(-0.0035646311345987206), np.float64(-0.5599786931647115), np.float64(-0.16927667510289623), np.float64(0.14435814786022258), np.float64(-0.6312335127672786), np.float64(0.07477309464080459), np.float64(0.0499624643000637), np.float64(-0.34254088615646333), np.float64(-0.020107524433700598), np.float64(-0.007001666603524885), np.float64(-0.5191956164208634), np.float64(0.01495639036278848), np.float64(-0.015696644601103833)]
params1=[np.float64(-0.4414919047525303), np.float64(-0.09039724002652984), np.float64(-0.283698445053175), np.float64(0.24175876779720587), np.float64(-0.5992479899019154), np.float64(-0.36565439126617866), np.float64(-1.0829581157114148), np.float64(-0.42646721817926453), np.float64(0.36358753971116287), np.float64(-0.8609737847245619), np.float64(-0.30122987575128984), np.float64(0.8893130045172475), np.float64(0.2502926924439254), np.float64(-0.016074548946125474), np.float64(-0.17534323050300613), np.float64(-0.2654137257006735), np.float64(-0.08266061906946941), np.float64(-0.019545987619465), np.float64(-0.7717717213862112), np.float64(0.3148082610062352), np.float64(-1.61523915158742), np.float64(-1.376012628594209), np.float64(0.32800054162354586), np.float64(-0.16324187889573794), np.float64(-0.7360168647479033), np.float64(0.3116749839747447), np.float64(1.0110607105360856), np.float64(0.37793014899819954), np.float64(-0.03313575396586794), np.float64(0.2647077436012146), np.float64(-1.3924848933176779), np.float64(0.08143730628761285), np.float64(0.10357840772268284), np.float64(-0.34180661975435556), np.float64(0.43416042713577263), np.float64(-0.14018675583092857), np.float64(0.3175897105698516), np.float64(-0.04937793629504618), np.float64(0.1461635151274783), np.float64(-0.7036607184026928), np.float64(0.2372637719834694), np.float64(0.04847886614282513), np.float64(-0.44849847037791507), np.float64(-0.25437717650470276), np.float64(-0.39224622567164624), np.float64(-0.238711463007887), np.float64(-0.01763856583242864), np.float64(-0.03279416145875986)]

params0_reshaped=np.array(params0).reshape((len(params0)//6,6))
params1_reshaped=np.array(params1).reshape((len(params1)//6,6))

gaussian_nonlincoeffs=np.concatenate((params0_reshaped[:,:4],params1_reshaped[:,:4]))
gaussian_nonlincoeffs=list(gaussian_nonlincoeffs)
#gaussian_nonlincoeffs=[]
initlen=int(sys.argv[5])
if initlen==20:
    gaussian_nonlincoeffs=[array([ 1.73419,  0.11343, -1.27528,  0.20927]), array([ 0.58529, -0.10182,  0.7117 , -1.76771]), array([ 2.11929,  2.17511, -0.88734, -1.16533]), array([ 1.29685,  0.04576, -0.1202 , -0.17472]), array([ 1.62669,  0.90041, -0.96891, -0.1685 ]), array([ 0.6496 , -0.11852, -0.91406,  0.3228 ]), array([ 0.49845,  0.03352, -0.13568, -0.78859]), array([-0.54314,  0.17441,  0.32153, -0.12239]), array([1.90321, 1.51622, 0.24345, 1.22702]), array([1.3594 , 0.2869 , 0.15527, 0.46575]), array([ 1.5183 , -0.10313, -0.91311, -0.04304]), array([ 0.5269 , -0.08562,  0.34713, -1.45937]), array([ 0.75832,  0.35972, -0.7476 , -1.51205]), array([ 1.13422,  0.20182,  0.17226, -0.24265]), array([ 1.4865 ,  0.50411, -1.10446, -0.51211]), array([ 0.4632 , -0.04265, -0.92913,  0.11099]), array([ 0.41095, -0.02081, -0.06822, -0.23986]), array([-0.76367,  0.09731,  0.46446, -0.79924]), array([ 0.24858, -0.00625,  0.41402, -0.16231]), array([-0.27584,  0.00092,  0.32731,  0.08046])]
elif initlen==14:
    gaussian_nonlincoeffs=[array([-1.87441,  0.06869, -0.01303, -1.16114]), array([-1.13952,  0.1118 ,  0.00996, -1.14995]), array([-0.66852,  0.15975,  0.80609, -0.15962]), array([ 0.50658, -0.00505,  0.04311, -0.72989]), array([-0.53134, -0.10374,  0.11515, -0.70682]), array([-0.38064,  0.0098 ,  0.01496, -0.71111]), array([-0.44047, -0.08777, -0.27224,  0.2496 ]), array([-1.07369, -0.38298,  0.366  , -0.86346]), array([ 0.24886, -0.01687, -0.17751, -0.27855]), array([-0.77583,  0.30321, -1.61006, -1.38048]), array([-0.75606,  0.30882,  1.03027,  0.38283]), array([-1.3378 ,  0.10531,  0.15715, -0.36631]), array([ 0.3179 , -0.04544,  0.14781, -0.69072]), array([ 0.51938, -0.25733, -0.40263, -0.27263])]
else:
    raise ValueError("initlen not supported")
n_extra=int(sys.argv[1])
print(initlen,n_extra)
if initlen==0:
    pos_list=np.linspace(-14,14,n_extra)
else:
    if typer=="linear" or typer=="basis":
        pos_list=np.concatenate([np.linspace(-14,-4,n_extra//2),np.linspace(4,14,n_extra//2)])
    else:
        pos_list=np.concatenate([np.linspace(-3,-8,n_extra//2),np.linspace(3,8,n_extra//2)])
for k in range(len(pos_list)):
    params=[1/sqrt(2),0,0,pos_list[k]]
    gaussian_nonlincoeffs.append(params)

gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs)
print(gaussian_nonlincoeffs.shape)
num_gauss=gaussian_nonlincoeffs.shape[0]
def gauss(x,a,b,p,q):
    bredde=(a**2 + 1j*b)
    qminx=q-x
    return abs(a)/sqrt(pi/2)*np.exp(-qminx*(1j*p + bredde*qminx))
def minus_half_laplacian(x,a,b,p,q):
    bredde=(a**2 + 1j*b)
    qminx=q-x
    return (bredde - 2.0*(0.5j*p + bredde*qminx)**2)*gauss(x,a,b,p,q)
@jit(nopython=True, fastmath=True,cache=False)
def gauss_and_minushalflaplacian(x, a, b, p, q):
    bredde = a**2 + 1j*b
    qminx = q - x
    jp=1j*p
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + bredde*qminx))
    minus_half_laplace = (bredde - 0.5 * (jp + 2*bredde*qminx)**2) * gaussval
    return gaussval, minus_half_laplace
@jit(nopython=True, fastmath=True,cache=False)

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


@jit(nopython=True, fastmath=True,cache=False)
def setupfunctions(gaussian_nonlincoeffs,points):
    if gaussian_nonlincoeffs.ndim==1:
        num_gauss=1
    else:
        num_gauss = len(gaussian_nonlincoeffs)
    functions = np.empty((num_gauss, len(points)), dtype=np.complex128)
    minus_half_laplacians = np.empty((num_gauss, len(points)), dtype=np.complex128)
    if gaussian_nonlincoeffs.ndim==1:
        avals=[gaussian_nonlincoeffs[0]]
        bvals=[gaussian_nonlincoeffs[1]]
        pvals=[gaussian_nonlincoeffs[2]]
        qvals=[gaussian_nonlincoeffs[3]]
    else:
        avals=gaussian_nonlincoeffs[:,0]
        bvals=gaussian_nonlincoeffs[:,1]
        pvals=gaussian_nonlincoeffs[:,2]
        qvals=gaussian_nonlincoeffs[:,3]
    for i in range(num_gauss):
        funcvals, minus_half_laplacian_vals = gauss_and_minushalflaplacian(points, avals[i], bvals[i], pvals[i], qvals[i])

        #funcvals, minus_half_laplacian_vals = gauss_and_minushalflaplacian(points, a_val_i, b_val_i, p_val_i, q_val_i)
        functions[i] = funcvals
        minus_half_laplacians[i] = minus_half_laplacian_vals
    
    return functions, minus_half_laplacians
@jit(nopython=True, fastmath=True,cache=False)
def setupfunctionsandDerivs(gaussian_nonlincoeffs,points):
    if gaussian_nonlincoeffs.ndim==1:
        num_gauss=1
    else:
        num_gauss = len(gaussian_nonlincoeffs)
    functions = np.empty((num_gauss, len(points)), dtype=np.complex128)
    minus_half_laplacians = np.empty((num_gauss, len(points)), dtype=np.complex128)
    aderiv_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    bderiv_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    pderiv_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    qderiv_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    aderiv_kin_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    bderiv_kin_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    pderiv_kin_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    qderiv_kin_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    if gaussian_nonlincoeffs.ndim==1:
        avals=[gaussian_nonlincoeffs[0]]
        bvals=[gaussian_nonlincoeffs[1]]
        pvals=[gaussian_nonlincoeffs[2]]
        qvals=[gaussian_nonlincoeffs[3]]
    else:
        avals=gaussian_nonlincoeffs[:,0]
        bvals=gaussian_nonlincoeffs[:,1]
        pvals=gaussian_nonlincoeffs[:,2]
        qvals=gaussian_nonlincoeffs[:,3]
    for i in range(num_gauss):
        funcvals, minus_half_laplacian_vals,da,db,dp,dq,dTa,dTb,dTp,dTq = gauss_and_minushalflaplacian_and_derivs(points, avals[i], bvals[i], pvals[i], qvals[i])

        functions[i] = funcvals
        minus_half_laplacians[i] = minus_half_laplacian_vals
        aderiv_funcs[i]=da
        bderiv_funcs[i]=db
        pderiv_funcs[i]=dp
        qderiv_funcs[i]=dq
        aderiv_kin_funcs[i]=dTa
        bderiv_kin_funcs[i]=dTb
        pderiv_kin_funcs[i]=dTp
        qderiv_kin_funcs[i]=dTq
    
    return (functions, minus_half_laplacians, aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, 
            aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs)
potential_grid=calculate_potential(Z_list,R_list,alpha,points)


onebody_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)
overlap_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)

wT=weights.T
def calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid,wT):
    for i in range(num_gauss):
        i_conj=np.conj(functions[i])
        for j in range(i,num_gauss):
            integrand_minus_half_laplace=i_conj*minus_half_laplacians[j]
            integrand_overlap=i_conj*functions[j]
            integrand_potential=integrand_overlap*potential_grid
            overlap_integraded=wT@integrand_overlap
            ij_integrated=wT@integrand_minus_half_laplace+wT@integrand_potential
            onebody_matrix[i,j]=ij_integrated
            onebody_matrix[j,i]=np.conj(ij_integrated)
            overlap_matrix[i,j]=overlap_integraded
            overlap_matrix[j,i]=np.conj(overlap_integraded)
    return onebody_matrix,overlap_matrix
e_e_grid=e_e_interaction(points)
@jit(nopython=True,fastmath=True,cache=False)
def calculate_twobody_integrals_numba(functions, e_e_grid, weights, num_gauss):
    twobody_integrals = np.zeros((num_gauss,num_gauss,num_gauss,num_gauss), dtype=np.complex128)
    
    # Precompute conjugated functions and cross products
    cross_functions = np.zeros((num_gauss, num_gauss, len(functions[0])), dtype=np.complex128)
    weighted_e_e_grid = e_e_grid * weights[:, np.newaxis]
    conj_functions=np.conj(functions)
    for i in range(num_gauss):
        for j in range(num_gauss):
            cross_functions[i, j] = conj_functions[i] * functions[j]
    for i in range(num_gauss):
        for k in range(i+1):
            index_ik=i*num_gauss+k
            ik_e_contr = np.sum(cross_functions[i, k][:, np.newaxis] * weighted_e_e_grid, axis=0)
            
            for j in range(num_gauss):
                for l in range(num_gauss):
                    index_jl=j*num_gauss+l
                    if index_ik<index_jl:
                        continue
                    jl_grid = cross_functions[j, l]
                    val=np.sum(jl_grid * ik_e_contr * weights)
                    cval=np.conj(val)
                    twobody_integrals[i, k, j, l] = val
                    twobody_integrals[j, l, i, k] = val

                    twobody_integrals[k, i, l, j] = cval
                    twobody_integrals[l, j, k, i] = cval


    return twobody_integrals
def restricted_hartree_fock(S, onebody, twobody, num_electrons, max_iterations=1, convergence_threshold=1e-9,C_init=None):
    """
    Perform Restricted Hartree-Fock (RHF) calculation for complex matrices with physicist's notation for two-body integrals.
    
    Args:
    S: Complex overlap matrix
    onebody: Complex one-body integrals
    twobody: Complex two-body integrals in physicist's notation (mu nu | lambda sigma)
    num_electrons: Number of electrons in the system
    max_iterations: Maximum number of SCF iterations
    convergence_threshold: Convergence criterion for energy difference
    
    Returns:
    E: Final Hartree-Fock energy
    C: Orbital coefficients
    F: Final Fock matrix
    """
    num_basis = S.shape[0]
    num_occupied = num_electrons // 2

    if num_electrons % 2 != 0:
        raise ValueError("RHF requires an even number of electrons.")

    # Step 1: Orthogonalize the basis (S^-1/2)
    s_eigenvalues, s_eigenvectors = linalg.eigh(S+1e-10*np.eye(S.shape[0]))
    X = linalg.inv(linalg.sqrtm(S+lambd*np.eye(S.shape[0])))
    #print(s_eigenvalues)

    # Step 2: Initial guess for density matrix
    if C_init is not None:
        C = C_init
    else:
        F = X.conj().T @ onebody @ X

        epsilon, C = linalg.eigh(F)
        C = X @ C
    P = 2*np.einsum("mj,vj->mv", C[:,:num_occupied], C.conj()[:,:num_occupied])
    E_old = 0
    for iteration in range(max_iterations):
        # Step 3: Build Fock matrix
        J = np.einsum('mnsl,ls->mn', twobody, P)
        K = np.einsum('mlsn,ls->mn', twobody, P)
        F = onebody+J - 0.5*K
        # Step 4: Calculate energy
        
        E=0.5*np.einsum("mn,nm->",P,F+onebody)
        if abs(E - E_old) < convergence_threshold:
            #print(f"Convergence reached at iteration {iteration + 1}")
            break

        # Step 5: Solve eigenvalue problem
        F_prime = X.conj().T @ F @ X
        epsilon, C_prime = linalg.eigh(F_prime)
        C = X @ C_prime[:,:num_occupied]
        epsilon=epsilon[:num_occupied]
        # Step 6: Form new density matrix
        P = 2*np.einsum("mj,vj->mv", C[:,:num_occupied], C.conj()[:,:num_occupied])

        E_old = E
    else:
        print(f"Warning: Reached maximum iterations ({max_iterations}) without converging.")

    return E, C, F,epsilon
def calculate_energy(gaussian_nonlincoeffs,return_all=False,C_init=None,maxiter=20):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    onebody_matrix,overlap_matrix=calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid,wT)
    twobody_integrals=calculate_twobody_integrals_numba(np.ascontiguousarray(functions), e_e_grid, weights, num_gauss)
    repulsion_contribution=0
    for i in range(len(Z_list)):
        for j in range(i+1,len(Z_list)):
            repulsion_contribution+=Z_list[i]*Z_list[j]/np.abs(R_list[i]-R_list[j])
    E,C,F,epsilon=restricted_hartree_fock(overlap_matrix,onebody_matrix,twobody_integrals,4,C_init=C_init,max_iterations=maxiter)
    Efinal=float(E+repulsion_contribution)
    print(Efinal)
    if return_all:
        print("Returning all")
        return Efinal,C,epsilon
    return Efinal

def make_orbitals_old(C,gaussian_nonlincoeffs):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    orbitals=[]
    nbasis=C.shape[0]
    norbs=C.shape[1]
    for i in range(norbs):
        orbital=np.zeros_like(points,dtype=np.complex128)
        for j in range(nbasis):
            orbital+=C[j,i]*functions[j]
        orbitals.append(orbital)
    return np.array(orbitals)
def make_orbitals_from_functions(C,functions):
    orbitals=[]
    nbasis=C.shape[0]
    norbs=C.shape[1]
    for i in range(norbs):
        orbital=np.zeros_like(points,dtype=np.complex128)
        for j in range(nbasis):
            orbital+=C[j,i]*functions[j]
        orbitals.append(orbital)
    return np.array(orbitals)
def make_orbitals(C,gaussian_nonlincoeffs):
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs.reshape((num_gauss,4)),points)
    return make_orbitals_numba(C,gaussian_nonlincoeffs,functions)
@jit(nopython=True,fastmath=True,cache=False)
def make_orbitals_numba(C,gaussian_nonlincoeffs,functions):
    nbasis=C.shape[0]
    norbs=C.shape[1]
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    orbitals=np.zeros((norbs,len(points)),dtype=np.complex128)
    for i in range(norbs):
        orbital=np.zeros_like(points,dtype=np.complex128)
        for j in range(nbasis):
            orbital+=C[j,i]*functions[j]
        orbitals[i]=orbital
    return orbitals

def calculate_Fgauss_old(fockOrbitals,gaussian_nonlincoeffs,num_gauss,time_dependent_potential=None):
    nFock=len(fockOrbitals)
    weighted_e_e_grid = e_e_grid * weights[:, np.newaxis]
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    Fgauss=np.zeros_like(functions)
    for i in range(num_gauss): 
        #For each gaussian
        #one body term
        Fgauss[i]+=minus_half_laplacians[i] #minus_half_laplacian term
        Fgauss[i]+=potential_grid*functions[i] #Potential term
        if time_dependent_potential is not None:
            Fgauss[i]+=time_dependent_potential*functions[i]
        for j in range(nFock):
            Fgauss[i]+=2*np.einsum("i,ij,i->j",np.conj(fockOrbitals[j]),weighted_e_e_grid,(fockOrbitals[j]))*functions[i] #Coulomb term
            Fgauss[i]-=1*np.einsum("i,ij,i->j",np.conj(fockOrbitals[j]),weighted_e_e_grid,(functions[i]))*fockOrbitals[j] #Coulomb term
    return Fgauss
def calculate_Fgauss(fockOrbitals,gaussian_nonlincoeffs,num_gauss,time_dependent_potential=None):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    return calculate_Fgauss_fast(np.array(fockOrbitals),num_gauss,time_dependent_potential,np.array(functions),np.array(minus_half_laplacians))
weighted_e_e_grid = e_e_grid * weights[:, np.newaxis]

@jit(nopython=True,fastmath=True,cache=False)
def calculate_Fgauss_fast(fockOrbitals,num_gauss,time_dependent_potential,functions,minus_half_laplacians):
    nFock=len(fockOrbitals)
    Fgauss=np.zeros_like(functions)
    Fgauss+=minus_half_laplacians
    if time_dependent_potential is not None:
        potential_term = potential_grid + time_dependent_potential
    else:
        potential_term = potential_grid
    Fgauss+=potential_term*functions
    coulomb_terms=np.zeros((nFock,len(points)),dtype=np.complex128)
    fock_orbitals_conj=np.conj(fockOrbitals)
    for j in range(nFock):
        coulomb_terms[j]=np.dot(fock_orbitals_conj[j] * fockOrbitals[j], weighted_e_e_grid)
    for i in range(num_gauss):
        for j in range(nFock):
            # Coulomb term (equivalent to first einsum)
            Fgauss[i] += 2 * coulomb_terms[j] * functions[i]

            # Exchange term (equivalent to second einsum)
            exchange_term = np.dot(fock_orbitals_conj[j] * functions[i], weighted_e_e_grid)
            Fgauss[i] -= exchange_term * fockOrbitals[j]
    return Fgauss
@jit(nopython=True,fastmath=True,cache=False)
def calculate_Ftimesorbitals(orbitals,FocktimesGauss):
    nbasis=orbitals.shape[0]
    norbs=orbitals.shape[1]
    FockOrbitals=np.zeros((norbs,len(points)),dtype=np.complex128)
    for i in range(norbs):
        FockOrbital=np.zeros_like(points,dtype=np.complex128)
        for j in range(nbasis):
            FockOrbital+=orbitals[j,i]*FocktimesGauss[j]
        FockOrbitals[i]=FockOrbital
    return FockOrbitals
"""
gaussian_nonlincoeffs=gaussian_nonlincoeffs.flatten()

E,C,epsilon=calculate_energy(gaussian_nonlincoeffs,return_all=True)

best_orbitals=make_orbitals(C,gaussian_nonlincoeffs)

FockGauss=calculate_Fgauss(gaussian_nonlincoeffs=gaussian_nonlincoeffs,num_gauss=num_gauss,fockOrbitals=best_orbitals)

Fock_times_Orbitals=calculate_Ftimesorbitals(C,FockGauss)

orn=1   
for orb in [0,1]:
    plt.plot(points,np.abs(best_orbitals[orb]*epsilon[orb]),label="|Basis F*psi(%d)|"%orb)
    plt.plot(points,np.abs(Fock_times_Orbitals[orb]),label="|Exact F*psi(%d)|"%orb)
    plt.plot(points,np.abs(best_orbitals[orb]*epsilon[orb]-Fock_times_Orbitals[orb]),label="diff")
plt.legend()
print("Energy: ",E)
plt.savefig('Fpsi.png', dpi=500)

plt.show()
for k in range(30):
    sol=minimize(calculate_energy,gaussian_nonlincoeffs,args=(False,C,30),method='Powell',options={'maxiter':50,"eps":5e-9})
    nonlin_params=sol.x
    gaussian_nonlincoeffs=nonlin_params
print(list(nonlin_params.reshape((num_gauss,4))))
print(len(nonlin_params)//4)
sys.exit(0)
"""
def calculate_Fock_and_overlap(C,gaussian_nonlincoeffs,time_dependent_potential=None):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    if time_dependent_potential is not None:
        onebody_matrix,overlap_matrix=calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid+time_dependent_potential,wT)
    else:
        onebody_matrix,overlap_matrix=calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid,wT)
    twobody_integrals=calculate_twobody_integrals_numba(np.ascontiguousarray(functions), e_e_grid, weights, num_gauss)
    P = 2*np.einsum("mj,vj->mv", C, C.conj())
    E_old = 0
    J = np.einsum('mnsl,ls->mn', twobody_integrals, P)
    K = np.einsum('mlsn,ls->mn', twobody_integrals, P)
    F = onebody_matrix+J - 0.5*K
    return F,overlap_matrix
def calculate_Fock_and_overlap_from_functions(C,functions,minus_half_laplacians,twobody_integrals,time_dependent_potential=None):
    if time_dependent_potential is not None:
        onebody_matrix,overlap_matrix=calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid+time_dependent_potential,wT)
    else:
        onebody_matrix,overlap_matrix=calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid,wT)
    P = 2*np.einsum("mj,vj->mv", C, C.conj())
    J = np.einsum('mnsl,ls->mn', twobody_integrals, P)
    K = np.einsum('mlsn,ls->mn', twobody_integrals, P)
    F = onebody_matrix+J - 0.5*K
    return F,overlap_matrix

def calculate_x_expectation(C,gaussian_nonlincoeffs):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    orbitals=make_orbitals(C,gaussian_nonlincoeffs)
    x_expectation=0
    for i in range(orbitals.shape[0]):
        val=2*weights.T@(np.abs(orbitals[i])**2*points)
        x_expectation+=val
    return x_expectation    
def make_Cmat_and_nonlin_from_params(full_params,n_gauss,num_orbs):
    n_lin_per_orb=n_gauss*num_orbs
    real_coefficients=full_params[:n_lin_per_orb]
    imag_coefficients=full_params[n_lin_per_orb:2*n_lin_per_orb]
    new_lincoeff=real_coefficients+1j*imag_coefficients
    new_lincoeff=new_lincoeff.reshape((n_gauss,num_orbs))
    new_params=full_params[2*n_lin_per_orb:]
    return new_params,new_lincoeff
def make_Cmat_from_truncated_params(full_params,n_gauss,num_orbs):
    n_lin_per_orb=n_gauss*num_orbs
    real_coefficients=full_params[:n_lin_per_orb]
    imag_coefficients=full_params[n_lin_per_orb:2*n_lin_per_orb]
    new_lincoeff=real_coefficients+1j*imag_coefficients
    new_lincoeff=new_lincoeff.reshape((n_gauss,num_orbs))
    return new_lincoeff
class Rothe_evaluator:
    def __init__(self,old_params,old_lincoeff,time_dependent_potential,timestep,number_of_frozen_orbitals=0):
        """
        old_params: The parameters for the Gaussians from the previous iteration
        old_lincoeff: The linear coefficients for the Gaussians in the basis of the old ones, from the previous iteration
        time_dependent_potential: The time-dependent potential evaluated at the relevant time
        timestep: The timestep used in the propagation
        """
        self.nbasis=old_lincoeff.shape[0]
        self.norbs=old_lincoeff.shape[1]

        self.old_params=old_params
        self.old_lincoeff=old_lincoeff
        self.pot=time_dependent_potential
        self.dt=timestep
        self.nfrozen=number_of_frozen_orbitals
        self.params_frozen=old_params[:4*self.nfrozen]
        self.orbitals_that_represent_Fock=make_orbitals(self.old_lincoeff,self.old_params) #Orbitals that define the Fock operator; which are the old orbitals

        self.old_action=self.calculate_Adagger_oldOrbitals() #Essentially, the thing we want to approximate with the new orbitals
    def calculate_Adagger_oldOrbitals(self):
        
        fock_act_on_old_gauss=calculate_Fgauss(self.orbitals_that_represent_Fock,self.old_params,num_gauss=self.nbasis,time_dependent_potential=self.pot) #Act with the OLD Fock operator on the OLD Gaussians
        Fock_times_Orbitals=calculate_Ftimesorbitals(self.old_lincoeff,fock_act_on_old_gauss)
        rhs=self.orbitals_that_represent_Fock-1j*self.dt/2*Fock_times_Orbitals
        return rhs
    def calculate_rothe_error_nonlin_params(self,nonlin_params_unfrozen,n_gauss):
        old_action=self.old_action *sqrt_weights
        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions,minus_half_laplacians=setupfunctions(nonlin_params.reshape((num_gauss,4)),points)
        fock_act_on_new_gauss=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=self.nbasis,time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        X=functions+1j*self.dt/2*fock_act_on_new_gauss
        new_lincoeff=np.empty((n_gauss,self.norbs),dtype=np.complex128)
        X=X.T
        X = X * sqrt_weights.reshape(-1, 1)

        XTX = X.conj().T @ X
        I=np.eye(XTX.shape[0])
        rothe_error=0
        for orbital_index in range(old_action.shape[0]):
            Y=old_action[orbital_index]
            XTy = X.conj().T @ Y
            cvals=np.linalg.inv(XTX+ lambd * I)@XTy
            new_lincoeff[:,orbital_index]=cvals
            cX=X@cvals
            rothe_error+=np.linalg.norm(Y-cX)**2
        self.optimal_lincoeff=new_lincoeff
        return rothe_error
    def rothe_plus_gradient(self,nonlin_params_unfrozen,n_gauss):
        old_action=self.old_action *sqrt_weights
        gradient=np.zeros_like(nonlin_params_unfrozen)
        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions,minus_half_laplacians,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(nonlin_params.reshape((-1,4)),points)
        fock_act_on_new_gauss=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=self.nbasis,time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        function_derivs=[]
        kin_derivs=[]
        for i in range(self.nfrozen,len(aderiv_funcs)):
            function_derivs+=[aderiv_funcs[i],bderiv_funcs[i],pderiv_funcs[i],qderiv_funcs[i]]
            kin_derivs+=[aderiv_kin_funcs[i],bderiv_kin_funcs[i],pderiv_kin_funcs[i],qderiv_kin_funcs[i]]
        function_derivs=np.array(function_derivs)
        kin_derivs=np.array(kin_derivs)
        indices_random=np.random.choice(len(old_action[0]), len(old_action[0])//2, replace=False);multiplier=2
        indices_random=np.array(np.arange(len(old_action[0]))); multiplier=1
        X=functions+1j*self.dt/2*fock_act_on_new_gauss
        new_lincoeff=np.empty((n_gauss,self.norbs),dtype=np.complex128)
        old_action=old_action[:,indices_random]
        X=X.T
        Xderc=np.zeros_like(X)[indices_random,:]
        X = X * sqrt_weights.reshape(-1, 1)
        X=X[indices_random,:]
        X_dag=X.conj().T
        XTX =X_dag @ X
        I=np.eye(XTX.shape[0])
        rothe_error=0
        zs=np.zeros_like(old_action)
        invmats=[]
        #
        for orbital_index in range(old_action.shape[0]):
            Y=old_action[orbital_index]
            XTy = X_dag @ Y
            invmats.append(np.linalg.inv(XTX+ lambd * I))
            new_lincoeff[:,orbital_index]=invmats[-1]@XTy
            zs[orbital_index]=Y-X@new_lincoeff[:,orbital_index]
            rothe_error+=np.linalg.norm(zs[orbital_index])**2*multiplier
        self.optimal_lincoeff=new_lincoeff
        Fock_act_on_derivs=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=self.nbasis,time_dependent_potential=self.pot,
                                                    functions=np.array(function_derivs),minus_half_laplacians=np.array(kin_derivs))
        
        Xders=function_derivs+1j*self.dt/2*Fock_act_on_derivs
        
        Xders=Xders.T
        Xders = Xders * sqrt_weights.reshape(-1, 1)
        Xders=Xders[indices_random,:]
        for i in range(len(nonlin_params_unfrozen)):
            Xder=Xderc.copy()
            Xder[:,self.nfrozen+i//4]=Xders[:,i]
            Xder_dag=Xder.conj().T
            for orbital_index in range(old_action.shape[0]):
                Y=old_action[orbital_index]
                invmat=invmats[orbital_index]
                XTYder=Xder_dag @ Y
                XTY=X_dag @ Y
                matrix_der=-invmat@(X_dag@Xder+Xder_dag@X)@invmat
                cder=matrix_der@XTY+invmat@XTYder
                gradient[i]+=2*np.real(zs[orbital_index].conj().T@(-Xder@new_lincoeff[:,orbital_index]-X@cder))*multiplier
        return rothe_error,gradient
    def calculate_rothe_error(self,full_new_params,n_gauss):
        old_action=self.old_action
        #The first n_gauss*n_orbs parameters are the real parts of full_new_params
        #The next n_gauss*n_orbs parameters are the imaginary parts of full_new_params
        #The rest are the non-linear parameters
        linear_params=full_new_params[:2*n_gauss*self.norbs]
        nonlin_params=np.concatenate((self.params_frozen,full_new_params[2*n_gauss*self.norbs:]))
        full_new_params=np.concatenate((linear_params,nonlin_params))
        new_params,new_lincoeff=make_Cmat_and_nonlin_from_params(full_new_params,n_gauss,self.norbs)

        
        functions,minus_half_laplacians=setupfunctions(new_params.reshape((num_gauss,4)),points)
        fock_act_on_new_gauss=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=self.nbasis,time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        print(new_lincoeff)
        X=functions+1j*self.dt/2*fock_act_on_new_gauss
        for orbital_index in range(old_action.shape[0]):
            Y=old_action[orbital_index]
            cvals=np.linalg.lstsq(X.T,Y)[0]
            new_lincoeff[:,orbital_index]=cvals
        new_orbitals=make_orbitals(new_lincoeff,nonlin_params)
        Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)

        new_action=new_orbitals+1j*self.dt/2*Fock_times_Orbitals

        orbital_errors=abs(new_action-old_action)
        weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
        rothe_error=np.linalg.norm(weighted_errors)**2
        self.optimal_lincoeff=new_lincoeff
        return rothe_error


    def calculate_numerical_rothe_jacobian(self,full_new_params,n_gauss,eps=1e-7):
        jac=np.zeros_like(full_new_params)
        old_action=self.old_action
        linear_params=full_new_params[:2*n_gauss*self.norbs]
        nonlin_params=np.concatenate((self.params_frozen,full_new_params[2*n_gauss*self.norbs:]))
        full_new_params_for_later=np.concatenate((linear_params,nonlin_params))
        new_params,new_lincoeff=make_Cmat_and_nonlin_from_params(full_new_params_for_later,n_gauss,self.norbs)
        new_orbitals=make_orbitals(new_lincoeff,new_params)
        fock_act_on_new_gauss=fock_act_on_new_gauss_old=calculate_Fgauss(self.orbitals_that_represent_Fock,new_params,num_gauss=self.nbasis,time_dependent_potential=self.pot)
        Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)
        new_action=new_orbitals+1j*self.dt/2*Fock_times_Orbitals
        orbital_errors=abs(new_action-old_action)
        weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
        re=np.linalg.norm(weighted_errors)**2
        
        for i in range(len(full_new_params)):
            full_new_params_copy=full_new_params.copy()
            full_new_params_copy[i]+=eps
            linear_params=full_new_params_copy[:2*n_gauss*self.norbs]
            nonlin_params=np.concatenate((self.params_frozen,full_new_params_copy[2*n_gauss*self.norbs:]))
            full_new_params_for_later_copy=np.concatenate((linear_params,nonlin_params))
            new_params,new_lincoeff=make_Cmat_and_nonlin_from_params(full_new_params_for_later_copy,n_gauss,self.norbs)
            new_orbitals=make_orbitals(new_lincoeff,new_params)


            #Instead, I only update a small part of Fock_times_Orbitals and update the new action
            if i<n_gauss*2*self.norbs:
                fock_act_on_new_gauss=fock_act_on_new_gauss_old
            else:
                index_of_gaussian=(i-n_gauss*2*self.norbs)//4
                params_to_update=new_params[index_of_gaussian*4:(index_of_gaussian+1)*4]
                fock_act_on_new_gauss_updated=calculate_Fgauss(self.orbitals_that_represent_Fock,params_to_update,num_gauss=1,time_dependent_potential=self.pot)
                fock_act_on_new_gauss=fock_act_on_new_gauss_old.copy()
                fock_act_on_new_gauss[index_of_gaussian]=fock_act_on_new_gauss_updated
            Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)
            new_action=new_orbitals+1j*self.dt/2*Fock_times_Orbitals
            orbital_errors=abs(new_action-old_action)
            weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
            rothe_error_pdE=np.linalg.norm(weighted_errors)**2
            jac[i]=(rothe_error_pdE-re)/eps
        return jac
    def calculate_numerical_rothe_jacobian_nonlin_params(self,nonlin_params_unfrozen,n_gauss,eps=1e-7):
        jac=np.zeros_like(nonlin_params_unfrozen)
        old_action=self.old_action
        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions,minus_half_laplacians=setupfunctions(nonlin_params.reshape((num_gauss,4)),points)
        fock_act_on_new_gauss_old=fock_act_on_new_gauss=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=self.nbasis,time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        X=functions+1j*self.dt/2*fock_act_on_new_gauss
        new_lincoeff=np.empty((n_gauss,self.norbs),dtype=np.complex128)
        X=X.T
        X = X * sqrt_weights.reshape(-1, 1)
        XTX = X.conj().T @ X
        I=np.eye(XTX.shape[0])
        for orbital_index in range(old_action.shape[0]):
            Y=old_action[orbital_index] * sqrt_weights
            

            XTy = X.conj().T @ Y
            #cvals=np.linalg.inv(XTX + lambd * I)@XTy
            cvals=np.linalg.inv(XTX+ lambd * I)@XTy
            new_lincoeff[:,orbital_index]=cvals
      
        new_orbitals=make_orbitals(new_lincoeff,nonlin_params)
        Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)

        new_action=new_orbitals+1j*self.dt/2*Fock_times_Orbitals

        orbital_errors=abs(new_action-old_action)
        weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
        re=np.linalg.norm(weighted_errors)**2
        
        for i in range(len(nonlin_params_unfrozen)):
            new_params_copy=nonlin_params_unfrozen.copy()
            new_params_copy[i]+=eps
            new_functions=functions.copy()
            params_to_update=new_params_copy[(i//4)*4:(i//4+1)*4]
            all_nonlin_params_for_later=np.concatenate((self.params_frozen,new_params_copy))
            new_function,new_mh_laplacian=setupfunctions(params_to_update,points)
            fock_times_gauss_update=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=1,time_dependent_potential=self.pot,
                                                    functions=np.array(new_function),minus_half_laplacians=np.array(new_mh_laplacian))
            fock_act_on_new_gauss=fock_act_on_new_gauss_old.copy()
            fock_act_on_new_gauss[self.nfrozen+(i//4)]=fock_times_gauss_update
            new_functions[self.nfrozen+(i//4)]=new_function
            X=functions+1j*self.dt/2*fock_act_on_new_gauss
            new_lincoeff=np.empty((n_gauss,self.norbs),dtype=np.complex128)
            X=X.T
            X = X * sqrt_weights.reshape(-1, 1)
            XTX = X.conj().T @ X
            
            for orbital_index in range(old_action.shape[0]):
                Y=old_action[orbital_index] * sqrt_weights
                

                XTy = X.conj().T @ Y
               
                cvals=np.linalg.inv(XTX+ lambd * I)@XTy
                new_lincoeff[:,orbital_index]=cvals
            new_orbitals=make_orbitals(new_lincoeff,all_nonlin_params_for_later)
            Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)

            new_action=new_orbitals+1j*self.dt/2*Fock_times_Orbitals

            orbital_errors=abs(new_action-old_action)
            weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
            newre=np.linalg.norm(weighted_errors)**2
            jac[i]=(newre-re)/eps
        return jac

    def calculate_Rothe_error_fixed_basis(self,lin_params,n_gauss):
        new_params=self.old_params
        new_lincoeff=make_Cmat_from_truncated_params(lin_params,n_gauss,self.norbs)
        new_orbitals=make_orbitals(new_lincoeff,new_params)
        fock_act_on_new_gauss=calculate_Fgauss(self.orbitals_that_represent_Fock,new_params,num_gauss=self.nbasis,time_dependent_potential=self.pot)
        Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)
        new_action=new_orbitals+1j*self.dt/2*Fock_times_Orbitals
        orbital_errors=abs(new_action-self.old_action)
        weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
        rothe_error=np.linalg.norm(weighted_errors)**2
        return rothe_error
    def calculate_numerical_rothe_jacobian_fixedbasis(self,lin_params,n_gauss,eps=1e-8):
        jac=np.zeros_like(lin_params)
        old_action=self.old_action
        new_params=self.old_params
        new_lincoeff=make_Cmat_from_truncated_params(lin_params,n_gauss,self.norbs)
        new_orbitals=make_orbitals(new_lincoeff,new_params)
        fock_act_on_new_gauss=fock_act_on_new_gauss_old=calculate_Fgauss(self.orbitals_that_represent_Fock,new_params,num_gauss=self.nbasis,time_dependent_potential=self.pot)
        Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)
        new_action=new_orbitals+1j*self.dt/2*Fock_times_Orbitals
        orbital_errors=abs(new_action-old_action)
        weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
        re=np.linalg.norm(weighted_errors)**2
        
        for i in range(len(lin_params)):
            full_new_params_copy=lin_params.copy()
            full_new_params_copy[i]+=eps
            new_lincoeff=make_Cmat_from_truncated_params(lin_params,n_gauss,self.norbs)
            new_orbitals=make_orbitals(new_lincoeff,new_params)

            #Instead, I only update a small part of Fock_times_Orbitals and update the new action
            fock_act_on_new_gauss=fock_act_on_new_gauss_old
            Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)
            new_action=new_orbitals+1j*self.dt/2*Fock_times_Orbitals
            orbital_errors=abs(new_action-old_action)
            weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
            rothe_error_pdE=np.linalg.norm(weighted_errors)**2
            jac[i]=(rothe_error_pdE-re)/eps
        return jac
class Rothe_propagation:
    def __init__(self,params_initial,lincoeffs_initial,pulse,timestep,points,nfrozen=0,t=0):
        self.nbasis=lincoeffs_initial.shape[0]
        self.norbs=lincoeffs_initial.shape[1]
        self.pulse=pulse
        self.dt=timestep
        params_initial=params_initial.flatten()
        self.lincoeffs=lincoeffs_initial
        self.params=params_initial
        self.functions=None
        self.nfrozen=nfrozen
        self.adjustment=None
        self.full_params=np.concatenate((lincoeffs_initial.flatten().real,lincoeffs_initial.flatten().imag,params_initial))
        self.t=t
    def propagate_fixed_basis(self,t):
        initial_lincoeffs=self.lincoeffs
        initial_params=self.params
        frozen_initial_params=initial_params[:80]
        self.time_dependent_potential=self.pulse(t+self.dt/2)*points
        F,S=calculate_Fock_and_overlap(initial_lincoeffs,initial_params,time_dependent_potential=self.time_dependent_potential)
        Spf=S+1j*dt/2*F
        Smf=S-1j*dt/2*F
        C_linear_update=np.linalg.solve(Spf,Smf@initial_lincoeffs) #Initial guess for the linear coefficients in a basisj of the new Gaussians
        C_flat=C_linear_update.flatten()
        initial_full_new_params=np.concatenate((C_flat.real,C_flat.imag,initial_params[80:]))
    def propagate(self,t):
        initial_lincoeffs=self.lincoeffs
        initial_params=self.params
        self.time_dependent_potential=self.pulse(t+self.dt/2)*points
        initial_full_new_params=initial_params[4*self.nfrozen:]
        rothe_evaluator=Rothe_evaluator(initial_params,initial_lincoeffs,self.time_dependent_potential,dt,self.nfrozen)
        initial_rothe_error=rothe_evaluator.calculate_rothe_error_nonlin_params(initial_full_new_params,self.nbasis)
        start_params=initial_params[4*self.nfrozen:]
        maxiter=0
        if t>=2:
            maxiter=40
        ls=np.linspace(0,1,3)
        best=0
        if self.adjustment is not None:
            updated_res=[initial_rothe_error]
            dx=self.adjustment
            for i in ls[1:]:
                start_params_alt=initial_full_new_params+i*dx

                updated_re=rothe_evaluator.calculate_rothe_error_nonlin_params(start_params_alt,self.nbasis)
                updated_res.append(updated_re)
            best=np.argmin(updated_res)
            start_params_alt=initial_full_new_params+best*dx
        print("Old Rothe error, using change of %.1f: %e"%(ls[best],initial_rothe_error))
        #sol=minimize(rothe_evaluator.calculate_rothe_error_nonlin_params,start_params,jac=rothe_evaluator.calculate_numerical_rothe_jacobian_nonlin_params,args=(self.nbasis),method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':1e-7})
        #solution=sol.x
        #new_rothe_error=rothe_evaluator.calculate_rothe_error_nonlin_params(solution,self.nbasis)
        if initial_rothe_error<1.6e-8:
            maxiter=0
        solution,new_rothe_error=minimize_transformed_bonds(rothe_evaluator.rothe_plus_gradient,
                                                    start_params=start_params,
                                                    gradient=True,
                                                    num_gauss=len(initial_lincoeffs),
                                                    num_frozen=self.nfrozen,
                                                    num_orbitals=2,
                                                    multi_bonds=0,
                                                    maxiter=maxiter,
                                                    both=True)
        
        print("New Rothe error: %e"%new_rothe_error)
        C_flat=rothe_evaluator.optimal_lincoeff.flatten()
        #C_flat=initial_lincoeffs.flatten()
        linparams_new=np.concatenate((C_flat.real,C_flat.imag))
        self.full_params=concatenated_params=np.concatenate((linparams_new,initial_params[:4*self.nfrozen],solution))
        self.adjustment=solution-start_params
       
        self.params=np.concatenate((initial_params[:4*self.nfrozen],solution))
        self.lincoeffs=rothe_evaluator.optimal_lincoeff
        print(np.reshape(solution,(n_extra,4)))
    def propagate_linear(self,t):
        initial_lincoeffs=self.lincoeffs
        initial_params=self.params
        self.time_dependent_potential=self.pulse(t+self.dt/2)*points
        F,S=calculate_Fock_and_overlap(initial_lincoeffs,initial_params,time_dependent_potential=self.time_dependent_potential)
        Spf=S+1j*dt/2*F
        Smf=S-1j*dt/2*F
        C_linear_update=np.linalg.solve(Spf,Smf@initial_lincoeffs) #Initial guess for the linear coefficients in a basisj of the new Gaussians
        C_flat=C_linear_update.flatten()
        #C_flat=initial_lincoeffs.flatten()
        initial_full_new_params=np.concatenate((C_flat.real,C_flat.imag))
        rothe_evaluator=Rothe_evaluator(initial_params,initial_lincoeffs,self.time_dependent_potential,dt)
        dg=1e-8
        #sol=minimize(rothe_evaluator.calculate_Rothe_error_fixed_basis,
        #               initial_full_new_params,args=(num_gauss,),method='BFGS',options={'maxiter':10, "hess_inv0": hess_inv0})
        solution=initial_full_new_params
        new_rothe_error=rothe_evaluator.calculate_Rothe_error_fixed_basis(initial_full_new_params,num_gauss)
        print("New Rothe error: ",new_rothe_error)
        new_lincoeff=make_Cmat_from_truncated_params(initial_full_new_params,num_gauss,self.norbs)
        self.lincoeffs=new_lincoeff
    def propagate_basis(self,t):
        initial_lincoeffs=self.lincoeffs
        self.time_dependent_potential=self.pulse(t+self.dt/2)*points
        if self.functions is None:
                self.functions,self.minus_half_laplacians=setupfunctions(self.params.reshape((num_gauss,4)),points)
                self.twobody_integrals=calculate_twobody_integrals_numba(np.ascontiguousarray(self.functions), e_e_grid, weights, num_gauss)
        # Compute Fock and overlap matrices
        F, S = calculate_Fock_and_overlap_from_functions(self.lincoeffs, 
                                                         self.functions,self.minus_half_laplacians,
                                                         self.twobody_integrals, self.time_dependent_potential)

        #F,S=calculate_Fock_and_overlap(initial_lincoeffs,initial_params,time_dependent_potential=self.time_dependent_potential)
        Spf=S+1j*dt/2*F
        Smf=S-1j*dt/2*F
        C_linear_update=np.linalg.solve(Spf,Smf@initial_lincoeffs) #Initial guess for the linear coefficients in a basisj of the new Gaussians
        C_flat=C_linear_update.flatten()
        initial_full_new_params=np.concatenate((C_flat.real,C_flat.imag))
        new_lincoeff=make_Cmat_from_truncated_params(initial_full_new_params,num_gauss,self.norbs)
        self.lincoeffs=new_lincoeff

    def propagate_nsteps_linear(self,nsteps):
        x_expectations=[]
        for i in range(nsteps):
            self.propagate_linear(i*self.dt)
            print("Step %d done"%i)
            #F,S=calculate_Fock_and_overlap(self.lincoeffs,self.params,time_dependent_potential=self.time_dependent_potential)
            x_expectation=calculate_x_expectation(self.lincoeffs,self.params)
            x_expectations.append(x_expectation)
            print("Time %.3f, <x>: %.6f"%(i*self.dt,x_expectation))
            #self.plot_orbitals((i+1)*self.dt)
        return x_expectations
    def propagate_nsteps(self,Tmax):
        x_expectations=[]
        times=np.linspace(0,nsteps*self.dt,nsteps+1)
        filename="Rothe_wavefunctions%.3f_%d_%d.npz"%(E0,initlen,num_gauss)
        x_expectation=calculate_x_expectation(self.lincoeffs,self.params)
        if self.t==0:
            #Delete the file fith filename if it exists
            try:
                os.remove(filename)
            except:
                pass
            save_wave_function(filename, self.full_params, self.dt,x_expectation,self.t)
        while self.t<Tmax:
            self.propagate(self.t)
            self.t+=self.dt
            #F,S=calculate_Fock_and_overlap(self.lincoeffs,self.params,time_dependent_potential=self.time_dependent_potential)
            save_wave_function(filename, self.full_params, self.dt,x_expectation,self.t)
            x_expectation=calculate_x_expectation(self.lincoeffs,self.params)
            x_expectations.append(x_expectation)
            print("Time %.3f, <x>: %.6f"%(self.t,x_expectation))
            #self.plot_orbitals((i+1)*self.dt)
        return x_expectations
    def propagate_nsteps_basis(self,nsteps):
        x_expectations=[]
        for i in range(nsteps):
            self.propagate_basis(i*self.dt)
            print("Step %d done"%i)
            #F,S=calculate_Fock_and_overlap(self.lincoeffs,self.params,time_dependent_potential=self.time_dependent_potential)
            x_expectation=calculate_x_expectation(self.lincoeffs,self.params)
            x_expectations.append(x_expectation)
            print("Time %.3f, <x>: %.6f"%(i*self.dt,x_expectation))
            #self.plot_orbitals((i+1)*self.dt)
        return x_expectations

    def plot_orbitals(self,t):
        plt.figure()
        orbitals=make_orbitals(self.lincoeffs,self.params)
        for i in range(self.norbs):
            plt.plot(points,np.abs(orbitals[i]),label="|Orbital %d|"%i)
        plt.legend()
        plt.savefig("Oribtals_t=%.3f.png"%(t), dpi=200)
        plt.close()


def laserfield(E0, omega, td):
    """
    Sine-squared laser pulse.

    Args:
        t (float): Time.
        E0 (float): Maximum field strength.
        omega (float): Laser frequency.
        td (float): Duration of the laser pulse.
    """
    def field(t):
        return -E0 * np.sin(omega * t) * np.sin(t*np.pi / td) ** 2
    return field
time_dependent_potential=0.1*points #I. e. 0.1*x - very strong field

E,lincoeff_initial,epsilon=calculate_energy(gaussian_nonlincoeffs,return_all=True)
E0 = float(sys.argv[2])  # Maximum field strength
omega = 0.06075  # Laser frequency
t_c = 2 * np.pi / omega  # Optical cycle
n_cycles = 3
dt=0.05
td = n_cycles * t_c  # Duration of the laser pulse
tfinal = td  # Total time of the simulation
nsteps=int(tfinal/dt)
print(tfinal)
t=np.linspace(0,tfinal,1000)
fieldfunc=laserfield(E0, omega, td)
#plt.plot(t, fieldfunc(t))
#plt.show()
#propagate_linear_basis(gaussian_nonlincoeffs,lincoeff_initial,fieldfunc,dt,points,int(tfinal/dt))
#sys.exit(0)




x_expectation_t0=calculate_x_expectation(lincoeff_initial,gaussian_nonlincoeffs)
functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs.reshape((num_gauss,4)),points)

Tmax=310
if typer=="linear":
    rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,timestep=dt,points=points,nfrozen=initlen)

    x_expectations=rothepropagator.propagate_nsteps_linear(nsteps)
    x_expectations=[x_expectation_t0]+x_expectations
    plt.plot(np.linspace(0,nsteps*dt,nsteps+1),x_expectations)
    np.savez("Rothe_linearbasis_%.3f_%d.npz"%(E0,num_gauss),t=np.linspace(0,nsteps*dt,nsteps+1),x=x_expectations)
    plt.show()
elif typer=="basis":
    rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,timestep=dt,points=points,nfrozen=initlen)

    x_expectations=rothepropagator.propagate_nsteps_basis(nsteps)
    x_expectations=[x_expectation_t0]+x_expectations
    plt.plot(np.linspace(0,nsteps*dt,nsteps+1),x_expectations)
    np.savez("linearbasis_%.3f_%d.npz"%(E0,num_gauss),t=np.linspace(0,nsteps*dt,nsteps+1),x=x_expectations)
    plt.show()
elif typer=="nonlinear":
    
    filename="Rothe_wavefunctions%.3f_%d_%d.npz"%(E0,initlen,num_gauss)
    try:
        np.load(filename)
        times=np.load(filename)["times"]
        try:
            start_time=float(sys.argv[4])
        except:
            start_time=times[-1]
        closest_index = np.abs(times - start_time).argmin()

        tmax=times[closest_index]
        print(tmax)
        params=np.load(filename)["params"]
        xvals=np.load(filename)["xvals"]
        time_step=np.load(filename)["time_step"]
        np.savez(filename, params=params[:closest_index+1], time_step=time_step,times=times[:closest_index+1],xvals=xvals[:closest_index+1])
        ngauss=len(params[0])//(4+2*norbs)
        
        lincoeff_initial_real=params[closest_index][:ngauss*norbs]#.reshape((ngauss,norbs))
        lincoeff_initial_complex=params[closest_index][ngauss*norbs:ngauss*norbs*2]#.reshape((ngauss,norbs))
        lincoeff_initial=lincoeff_initial_real+1j*lincoeff_initial_complex
        lincoeff_initial=lincoeff_initial.reshape((ngauss,norbs))
        gaussian_nonlincoeffs=params[closest_index][ngauss*norbs*2:]
    except FileNotFoundError:
        tmax=0
    print(gaussian_nonlincoeffs.shape)
    rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,timestep=dt,points=points,nfrozen=initlen,t=tmax)
    x_expectations=rothepropagator.propagate_nsteps(Tmax)
    x_expectations=[x_expectation_t0]+x_expectations
    plt.plot(np.linspace(0,nsteps*dt,nsteps+1),x_expectations)
    np.savez("Rothe_%.3f_%d.npz"%(E0,num_gauss),t=np.linspace(0,nsteps*dt,nsteps+1),x=x_expectations)
    plt.show()