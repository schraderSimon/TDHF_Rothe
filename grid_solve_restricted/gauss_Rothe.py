import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numba
from numba import prange,jit
import sys
import os
#import sympy as sp
from scipy import linalg
import time
from numpy.polynomial.hermite import hermgauss

import warnings
warnings.filterwarnings("error", category=RuntimeWarning, message="invalid value encountered in arctanh")
#warnings.filterwarnings("error", category=RuntimeWarning, message="divide by zero encountered in arctanh")

from scipy.optimize import minimize
from numpy import array,sqrt, pi
from numpy import exp
from numpy import cosh, tanh, arctanh
#from sympy import *
from quadratures import *
from helper_functions import *
from optimizers import diis
np.set_printoptions(linewidth=300, precision=6, suppress=True, formatter={'float': '{:0.4e}'.format})
class ConvergedError(Exception):
    def __init__(self):
        pass
avals_min=1e-1
avals_max=2
bvals_min=-10
bvals_max=10
pvals_min=-20
pvals_max=20

def Coulomb(x, Z, x_c=0.0, alpha=1.0):
    """
    Coulomb potential in 1D.

    Args:
        x (np.ndarray): The grid points.
        Z (float): The nuclear charge.
        x_c (float): The nuclear position.
        a (float): The regularization parameter.
    """
    return -Z / np.sqrt((x - x_c) ** 2 + alpha)

class Molecule1D:
    def __init__(self, R=[0.0], Z=[1], alpha=1.0):
        """
        Molecular potential in 1D.

        Args:
            R (list): The nuclear positions.
            Z (list): The nuclear charges.
            alpha (float): The regularization parameter.
        """
        self.R_list = R
        self.Z_list = Z
        if alpha <= 0:
            raise ValueError("The regularization parameter must be positive.")
        self.alpha = alpha

    def __call__(self, x):
        if isinstance(x, float):
            potential = 0
        else:
            potential = np.zeros(len(x),dtype=np.complex128)
        for R, Z in zip(self.R_list, self.Z_list):
            potential += Coulomb(x, Z=Z, x_c=R, alpha=self.alpha)
        return potential


# Function to save or append data
def save_wave_function(filename, wave_function, time_step,xval,time,rothe_error,norms,ngauss,norbs):
    try:
        # Try to load the existing data
        existing_data = np.load(filename)
        params=existing_data['params']
        params=list(params)
        xvals=list(existing_data['xvals'])
        times=list(existing_data['times'])
        rothe_errors=list(existing_data['rothe_errors'])
        norms_list=list(existing_data['norms'])
        number_of_basis_functions=list(existing_data['nbasis'])
    except FileNotFoundError:
        params=[]
        xvals=[]
        times=[]
        rothe_errors=[]
        norms_list=[]
        number_of_basis_functions=[]
    if len(params)==0 or len(params[-1])==len(wave_function):
        pass
    elif len(params[-1])<len(wave_function):
        print("Wave function has wrong length")
        print(ngauss)
        for i in range(len(params)):
            params_i=np.zeros_like(wave_function)
            lincoeff_real=params[i][:(ngauss-1)*norbs]#.reshape((ngauss,norbs))
            lincoeff_complex=params[i][(ngauss-1)*norbs:(ngauss-1)*norbs*2]#.reshape((ngauss,norbs))
            gaussian_nonlincoeffs=params[i][(ngauss-1)*norbs*2:]
            params_i[:(ngauss-1)*norbs]=lincoeff_real
            params_i[ngauss*norbs:(2*ngauss-1)*norbs]=lincoeff_complex
            params_i[ngauss*norbs*2:-4]=gaussian_nonlincoeffs
            params_i[-4:]=[1,0,0,0]
            params[i]=params_i
    elif len(params[-1])>len(wave_function):
        print("Wave function has wrong length")
        for i in range(len(params)):
            ngauss_wrong=len(params[i])//(4+norbs*2)
            params_i=np.zeros_like(wave_function)
            lincoeff_real=params[i][:ngauss*norbs]
            lincoeff_complex=params[i][ngauss_wrong*norbs:(ngauss+ngauss_wrong)*norbs]
            gaussian_nonlincoeffs=params[i][ngauss_wrong*norbs*2:-4*(ngauss_wrong-ngauss)]
            params_i[:len(lincoeff_real)]=lincoeff_real
            params_i[len(lincoeff_real):len(lincoeff_real)+len(lincoeff_complex)]=lincoeff_complex
            params_i[len(lincoeff_real)+len(lincoeff_complex):]=gaussian_nonlincoeffs
            params[i]=params_i
    params.append(wave_function)
    xvals.append(xval)
    times.append(time)
    rothe_errors.append(rothe_error)
    norms_list.append(norms)
    number_of_basis_functions.append(ngauss)
    # Append new wave function at the current time step
    np.savez(filename, params=params, time_step=time_step,times=times,xvals=xvals,rothe_errors=rothe_errors,norms=norms_list,nbasis=number_of_basis_functions)
    print("Time: %.2f, Cumul R.E.: %.2e \n"%(times[-1],np.sum((rothe_errors))))
def cosine4_mask(x, a, b):
    a0=0.85*a
    b0=b*0.85
    returnval=np.zeros_like(x)
    left_absorber=np.cos(pi/2*(a0-x)/(a-a0))**(1/4)
    right_absorber=np.cos(pi/2*(b0-x)/(b0-b))**(1/4)
    returnval[(x<a0) & (x>a)]=left_absorber[(x<a0) & (x>a)]
    returnval[(x>b0) & (x<b)]=right_absorber[(x>b0) & (x<b)]
    returnval[(x<a) | (x>b)]=0
    returnval[(x>a0) & (x<b0)]=1
    return returnval

def minimize_hessian(error_function,start_params,num_gauss):
    params=start_params
    param_list=[]
    error_list=[]
    i_min=5
    i_max=15
    for i in range(i_max):
        error,grad,hess=error_function(params,num_gauss,hessian=True)
        diag_hess=np.diag(np.diag(hess))
        hess_inv=np.linalg.inv(hess+1000*diag_hess)
        paramsNew=params-np.dot(hess_inv,grad)
        #print("Iteration %d: Error: %.3e"%(i,error))
        params=paramsNew
        param_list.append(params)
        error_list.append(error)
        if i>=i_min:
            if error_list[i]/error_list[i-i_min]>0.99:
                break
    print("Niter: %d"%len(error_list))
    best_error=np.argmin(error_list)
    return param_list[best_error],error_list[best_error]
def minimize_transformed_bonds(error_function,start_params,multi_bonds=0.1,gtol=1e-9,maxiter=20,gradient=None,both=False,lambda_grad0=1e-8,hess_inv=None,scale="log",intervene=True):
    """
    Minimizes with min_max bonds as described in https://lmfit.github.io/lmfit-py/bounds.html
    """
    def transform_params(untransformed_params):
        newparams=np.zeros_like(untransformed_params)
        for i,param in enumerate(2*(untransformed_params-mins)/(maxs-mins)-1):
            if param>0.9999:
                newparams[i]=5
            elif param<-0.9999:
                newparams[i]=-5
            else:
                newparams[i]=arctanh(param)
        return newparams
        #return arcsin(2*(untransformed_params-mins)/(maxs-mins)-1)
    def untransform_params(transformed_params):
        return mins+(maxs-mins)/2*(1+tanh(transformed_params))
        #return mins+(maxs-mins)/2*(1+sin(transformed_params))
    def chainrule_params(transformed_params):
        coshvals=cosh(transformed_params)
        returnval= 0.5*(maxs-mins)/(coshvals**2)
        return returnval
        #return 0.5*(maxs-mins)*cos(transformed_params)

    def transformed_error(transformed_params):
        error=error_function(untransform_params(transformed_params))
        return error
    def transformed_gradient(transformed_params):
        orig_grad=gradient(untransform_params(transformed_params))
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
        error,orig_grad=error_function(untransformed_params)
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
    dp[0::4]/=2 #multi bonds for a is half of the other parameters as it is the most sensitive parameter
    range_nonlin=[0.01,0.1,0.3,0.3]*(len(start_params)//4)
    rangex=range_nonlin
    rangex=np.array(rangex)
    mins=start_params-rangex-dp*abs(start_params)
    maxs=start_params+rangex+dp*abs(start_params)
    mmins=np.zeros_like(mins)
    mmaxs=np.zeros_like(maxs)
    mmins[0::4]=avals_min
    mmaxs[0::4]=avals_max
    mmins[1::4]=bvals_min
    mmaxs[1::4]=bvals_max
    mmins[2::4]=pvals_min
    mmaxs[2::4]=pvals_max
    mmins[3::4]=muvals_min
    mmaxs[3::4]=muvals_max

    for i in range(len(mins)):
        if mins[i]<mmins[i]:
            mins[i]=mmins[i]
        if maxs[i]>mmaxs[i]:
            maxs[i]=mmaxs[i]
    for i,sp in enumerate(start_params):
        if abs(sp-mins[i])<1e-1:
            pass
            #start_params[i]=mins[i]+1e-2
        if abs(sp-maxs[i])<1e-2:
            pass
            #start_params[i]=maxs[i]-1e-2

    transformed_params=transform_params(start_params)
    transformed_params=np.real(np.nan_to_num(transformed_params))

    start=time.time()
    if both:
        err0,grad0=transform_error_and_gradient(transformed_params)
    else:
        grad0=transformed_gradient(transformed_params)
    if hess_inv is None:
        hess_inv0=np.eye(len(grad0))/np.linalg.norm(grad0)*1e2
        hess_inv0=np.diag(1/abs(grad0+lambda_grad0*np.array(len(grad0))))

    else:
        hess_inv0=hess_inv
    
    numiter=0
    f_storage=[]
    def callback_func(intermediate_result: scipy.optimize.OptimizeResult):
        if intervene:
            nonlocal numiter
            nonlocal f_storage
            nonlocal transformed_sol
            nonlocal minval
            transformed_sol=intermediate_result.x
            fun=intermediate_result.fun
            minval=fun
            if scale=="log":
                re=sqrt(np.exp(fun))
            else:
                re=sqrt(fun)
            f_storage.append(re)
            miniter=20
            compareto_opt=20
            compareto=compareto_opt if compareto_opt<miniter else miniter-1
            if  numiter>=miniter: #At least 30 iterations
                if f_storage[-1]/f_storage[-compareto]>0.999 and f_storage[-1]/f_storage[-compareto]<1:
                    raise ConvergedError
            numiter+=1

    converged=False
    try:
        
        if both is False:
            sol=minimize(transformed_error,transformed_params,jac=transformed_gradient,
                        method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':gtol},
                        callback=callback_func)
        else:
            sol=minimize(transform_error_and_gradient,transformed_params,jac=True,
                        method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':gtol},
                        callback=callback_func)
        transformed_sol = sol.x
        minval=sol.fun #Not really fun, I am lying here
        numiter=sol.nit
        """
        transformed_sol,numiter,minval=diis(transform_error_and_gradient,transformed_params)
        """
    except ConvergedError:
        converged=True
        """
        This means that the callback function tells me that the function is not decreasing anymore.
        The important values are updated in the callback function (self.transformed_sol and self.minval)
        so that this "error" can safely be ignored.
        """
        pass

    end=time.time()
    return untransform_params(transformed_sol), minval, end-start,numiter
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
def gauss(x,a,b,p,q):
    bredde = a**2 + 1j*b
    qminx = q - x
    jp=1j*p
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + bredde*qminx))
    return gaussval
def minus_half_laplacian(x,a,b,p,q):
    bredde=(a**2 + 1j*b)
    qminx=q-x
    return (bredde - 2.0*(0.5j*p + bredde*qminx)**2)*gauss(x,a,b,p,q)
@jit(nopython=True, fastmath=True,cache=True)
def gauss_and_minushalflaplacian(x, a, b, p, q):
    bredde = a**2 + 1j*b
    qminx = q - x
    jp=1j*p
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + bredde*qminx))
    minus_half_laplace = (bredde - 0.5 * (jp + 2*bredde*qminx)**2) * gaussval
    return gaussval, minus_half_laplace

@jit(nopython=True, fastmath=True,cache=True)
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
    tempval=0.5*jp + br_qminx
    tempvalsq=tempval**2
    brmttvsq=bredde - 2*tempvalsq
    b_inv=1/brmttvsq
    aderiv_kin=minus_half_laplace*(-asq*(qminx**2*(4*asq + 4*1j*b - 8*tempvalsq) + 16*qminx*tempval - 4) + brmttvsq)/(2*a)*b_inv
    bderiv_kin=minus_half_laplace*1j*(-qminx**2*brmttvsq - 4*qminx*tempval + 1)*b_inv
    pderiv_kin=minus_half_laplace*1j*(-jp - 2.0*br_qminx - qminx*brmttvsq)*b_inv
    qderiv_kin=minus_half_laplace*(-2.0*(2*asq + 2j*b)*tempval - (jp + 2*br_qminx)*brmttvsq)*b_inv
    return (gaussval, minus_half_laplace, aderiv, bderiv, pderiv, qderiv, 
            aderiv_kin, bderiv_kin, pderiv_kin, qderiv_kin)


@jit(nopython=True, fastmath=True,parallel=True)
def setupfunctions(gaussian_nonlincoeffs,points):
    if gaussian_nonlincoeffs.ndim==1:
        num_gauss=1
    else:
        num_gauss = len(gaussian_nonlincoeffs)
    functions = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    minus_half_laplacians = np.zeros((num_gauss, len(points)), dtype=np.complex128)
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
    for i in prange(num_gauss):
        indices_of_interest=np.where((np.abs(points-qvals[i])*avals[i])<6)
        funcvals, minus_half_laplacian_vals = gauss_and_minushalflaplacian(points[indices_of_interest], avals[i], bvals[i], pvals[i], qvals[i])

        functions[i][indices_of_interest] = funcvals
        minus_half_laplacians[i][indices_of_interest] = minus_half_laplacian_vals
    
    return functions, minus_half_laplacians
@jit(nopython=True, fastmath=True,parallel=True)
def setupfunctionsandDerivs(gaussian_nonlincoeffs,points):
    if gaussian_nonlincoeffs.ndim==1:
        num_gauss=1
    else:
        num_gauss = len(gaussian_nonlincoeffs)
    functions = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    minus_half_laplacians = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    aderiv_funcs = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    bderiv_funcs = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    pderiv_funcs = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    qderiv_funcs = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    aderiv_kin_funcs = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    bderiv_kin_funcs = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    pderiv_kin_funcs = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    qderiv_kin_funcs = np.zeros((num_gauss, len(points)), dtype=np.complex128)
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
    for i in prange(num_gauss):
        indices_of_interest=np.where((np.abs(points-qvals[i])*avals[i])<6)
        funcvals, minus_half_laplacian_vals,da,db,dp,dq,dTa,dTb,dTp,dTq = gauss_and_minushalflaplacian_and_derivs(points[indices_of_interest], avals[i], bvals[i], pvals[i], qvals[i])

        functions[i][indices_of_interest] = funcvals
        minus_half_laplacians[i][indices_of_interest] = minus_half_laplacian_vals
        aderiv_funcs[i][indices_of_interest]=da
        bderiv_funcs[i][indices_of_interest]=db
        pderiv_funcs[i][indices_of_interest]=dp
        qderiv_funcs[i][indices_of_interest]=dq
        aderiv_kin_funcs[i][indices_of_interest]=dTa
        bderiv_kin_funcs[i][indices_of_interest]=dTb
        pderiv_kin_funcs[i][indices_of_interest]=dTp
        qderiv_kin_funcs[i][indices_of_interest]=dTq
    
    return (functions, minus_half_laplacians, aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, 
            aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs)
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
def restricted_hartree_fock(S, onebody, twobody, num_electrons, max_iterations=1, convergence_threshold=1e-11,C_init=None):
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
    print(num_occupied)
    if num_electrons % 2 != 0:
        raise ValueError("RHF requires an even number of electrons.")

    # Step 1: Orthogonalize the basis (S^-1/2)
    s_eigenvalues, s_eigenvectors = linalg.eigh(S+lambd*np.eye(S.shape[0]))
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
    num_gauss=len(gaussian_nonlincoeffs.flatten())//4
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    onebody_matrix,overlap_matrix=calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid,wT)
    twobody_integrals=calculate_twobody_integrals_numba(np.ascontiguousarray(functions), e_e_grid, weights, num_gauss)
    repulsion_contribution=0
    for i in range(len(Z_list)):
        for j in range(i+1,len(Z_list)):
            repulsion_contribution+=Z_list[i]*Z_list[j]/np.abs(R_list[i]-R_list[j])
    n_elec=4
    if molecule=="LiH2":
        n_elec=8
    E,C,F,epsilon=restricted_hartree_fock(overlap_matrix,onebody_matrix,twobody_integrals,n_elec,C_init=C_init,max_iterations=maxiter)
    Efinal=float(E+repulsion_contribution)
    print(Efinal)
    if return_all:
        print("Returning all")
        return Efinal,C,epsilon
    return Efinal

def make_orbitals(C,gaussian_nonlincoeffs):
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs.reshape((C.shape[0],4)),points)
    return make_orbitals_numba(C,gaussian_nonlincoeffs,functions)

@jit(nopython=True,fastmath=True,cache=False)
def make_orbitals_numba(C,gaussian_nonlincoeffs,functions):
    nbasis=C.shape[0]
    norbs=C.shape[1]
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((nbasis,4))
    orbitals=np.zeros((norbs,len(points)),dtype=np.complex128)
    for i in range(norbs):
        orbital=np.zeros_like(points,dtype=np.complex128)
        for j in range(nbasis):
            orbital+=C[j,i]*functions[j]
        orbitals[i]=orbital
    return orbitals

def calculate_Fgauss(fockOrbitals,gaussian_nonlincoeffs,num_gauss,time_dependent_potential=None):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    return calculate_Fgauss_fast(np.array(fockOrbitals),num_gauss,time_dependent_potential,np.array(functions),np.array(minus_half_laplacians))

@jit(nopython=True,fastmath=True,cache=False)
def calculate_Fgauss_fast(fockOrbitals,num_gauss,time_dependent_potential,functions,minus_half_laplacians):
    nFock=len(fockOrbitals)
    #Fgauss=np.zeros_like(functions)
    Fgauss=minus_half_laplacians
    potential_term = potential_grid + time_dependent_potential
    Fgauss+=potential_term*functions
    coulomb_terms=np.empty((nFock,fockOrbitals.shape[1]),dtype=np.complex128)
    for j in range(nFock):
        coulomb_terms[j]=np.dot(np.conj(fockOrbitals[j]) * fockOrbitals[j], weighted_e_e_grid)
    fock_orbitals_conj=np.conj(fockOrbitals)
    #product=fock_orbitals_conj * fockOrbitals
    #sum_coulomb=2*np.sum(np.tensordot(product, weighted_e_e_grid, axes=([1], [0])),axis=0)
    #Fgauss+=sum_coulomb*functions
    for i in range(num_gauss):
        for j in range(nFock):
            exchange_term =(fock_orbitals_conj[j] * functions[i]).T@weighted_e_e_grid
            Fgauss[i] += 2 * coulomb_terms[j] *functions[i]-exchange_term * fockOrbitals[j]
    return Fgauss
@jit(nopython=True,fastmath=True,cache=False)
def calculate_Ftimesorbitals(orbitals,FocktimesGauss):
    nbasis=orbitals.shape[0]
    norbs=orbitals.shape[1]
    FockOrbitals=np.empty((norbs,len(points)),dtype=np.complex128)
    for i in range(norbs):
        FockOrbital=np.zeros_like(points,dtype=np.complex128)
        for j in range(nbasis):
            FockOrbital+=orbitals[j,i]*FocktimesGauss[j]
        FockOrbitals[i]=FockOrbital
    return FockOrbitals

def calculate_x_expectation(C,gaussian_nonlincoeffs):
    num_gauss=C.shape[0]
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
        self.f_frozen,self.fock_act_on_frozen_gauss=self.calculate_frozen_orbital_stuff()
        
    def calculate_Adagger_oldOrbitals(self):
        
        fock_act_on_old_gauss=calculate_Fgauss(self.orbitals_that_represent_Fock,self.old_params,num_gauss=self.nbasis,time_dependent_potential=self.pot) #Act with the OLD Fock operator on the OLD Gaussians
        Fock_times_Orbitals=calculate_Ftimesorbitals(self.old_lincoeff,fock_act_on_old_gauss)
        rhs=self.orbitals_that_represent_Fock-1j*self.dt/2*Fock_times_Orbitals
        return rhs
    def calculate_frozen_orbital_stuff(self):
        functions,minus_half_laplacians,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(self.params_frozen.reshape((-1,4)),points)
        fock_act_on_frozen_gauss=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions),time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        """
        function_derivs=[]
        kin_derivs=[]
        for i in range(self.nfrozen,len(aderiv_funcs)):
            function_derivs+=[aderiv_funcs[i],bderiv_funcs[i],pderiv_funcs[i],qderiv_funcs[i]]
            kin_derivs+=[aderiv_kin_funcs[i],bderiv_kin_funcs[i],pderiv_kin_funcs[i],qderiv_kin_funcs[i]]
        
        function_derivs=np.array(function_derivs)
        kin_derivs=np.array(kin_derivs)
        Fock_act_on_derivs=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(function_derivs),time_dependent_potential=self.pot,
                                                    functions=np.array(function_derivs),minus_half_laplacians=np.array(kin_derivs))
        """
        return functions,fock_act_on_frozen_gauss
    
    def rothe_plus_gradient(self,nonlin_params_unfrozen,hessian=False):
        old_action=self.old_action *sqrt_weights
        gradient=np.zeros_like(nonlin_params_unfrozen)
        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions_u,minus_half_laplacians_u,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(nonlin_params_unfrozen.reshape((-1,4)),points)
        fock_act_on_new_gauss=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions_u),time_dependent_potential=self.pot,
                                                    functions=np.array(functions_u),minus_half_laplacians=np.array(minus_half_laplacians_u))
        functions=np.concatenate((self.f_frozen,functions_u))
        fock_act_on_functions=np.concatenate((self.fock_act_on_frozen_gauss,fock_act_on_new_gauss))
        function_derivs=[]
        kin_derivs=[]
        for i in range(len(aderiv_funcs)):
            function_derivs+=[aderiv_funcs[i],bderiv_funcs[i],pderiv_funcs[i],qderiv_funcs[i]]
            kin_derivs+=[aderiv_kin_funcs[i],bderiv_kin_funcs[i],pderiv_kin_funcs[i],qderiv_kin_funcs[i]]
        function_derivs=np.array(function_derivs)
        kin_derivs=np.array(kin_derivs)
        indices_random=np.random.choice(len(old_action[0]), len(old_action[0])//2, replace=False);multiplier=2
        indices_random=np.array(np.arange(len(old_action[0]))); multiplier=1
        X=functions+1j*self.dt/2*fock_act_on_functions
        n_gridpoints=X.shape[1]
        n_params=len(nonlin_params_unfrozen)
        new_lincoeff=np.empty((self.nbasis,self.norbs),dtype=np.complex128)
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
                                                    num_gauss=len(function_derivs),time_dependent_potential=self.pot,
                                                    functions=np.array(function_derivs),minus_half_laplacians=np.array(kin_derivs))
        #Fock_act_on_derivs=np.concatenate((self.Fock_act_on_frozen_derivs,Fock_act_on_derivs))
        #function_derivs=np.concatenate((self.f_frozen_derivs,function_derivs))
        Xders=function_derivs+1j*self.dt/2*Fock_act_on_derivs
        
        Xders=Xders.T
        Xders = Xders * sqrt_weights.reshape(-1, 1)
        Xders=Xders[indices_random,:]
        gradvecs=np.zeros((n_params,n_gridpoints),dtype=np.complex128)
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
                gradvec=(-Xder@new_lincoeff[:,orbital_index]-X@cder)
                gradient[i]+=2*np.real(zs[orbital_index].conj().T@gradvec)*multiplier
                gradvecs[i]+=gradvec
        
        gradvecs=np.array(gradvecs)
        if hessian:
            hessian_val=np.real(2*gradvecs@np.conj(gradvecs).T)
        if hessian:
            return rothe_error,gradient,hessian_val
        else:
            return rothe_error,gradient

    def rothe_error_oneremoved(self,nonlin_params_unfrozen):
        old_action=self.old_action *sqrt_weights
        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions,minus_half_laplacians,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(nonlin_params.reshape((-1,4)),points)
        fock_act_on_new_gauss=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions),time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        X=functions+1j*self.dt/2*fock_act_on_new_gauss
        new_lincoeff=np.empty((self.nbasis,self.norbs),dtype=np.complex128)
        old_action=old_action
        X=X.T
        X = X * sqrt_weights.reshape(-1, 1)
        X_dag=X.conj().T
        XTX =X_dag @ X
        I=np.eye(XTX.shape[0])
        rothe_error=0
        zs=np.zeros_like(old_action)
        invmats=[]
        rothe_error_gaussian_removed=np.zeros(len(nonlin_params_unfrozen)//4)
        for orbital_index in range(old_action.shape[0]):
            Y=old_action[orbital_index]
            XTy = X_dag @ Y
            invmats.append(np.linalg.inv(XTX+ lambd * I))
            new_lincoeff[:,orbital_index]=invmats[-1]@XTy
            zs[orbital_index]=Y-X@new_lincoeff[:,orbital_index]
            rothe_error+=np.linalg.norm(zs[orbital_index])**2
            for i in range(len(self.params_frozen)//4,len(nonlin_params)//4):
                mask = np.arange(X.shape[1]) != i
                X_masked=X[:,mask]
                X_dag_masked=X_masked.conj().T
                XTX_masked=X_dag_masked@X_masked
                I_masked=np.eye(XTX_masked.shape[0])
                invmat_masked=np.linalg.inv(XTX_masked+ lambd * I_masked)
                new_lincoeff_masked=invmat_masked@X_dag_masked@Y
                zs_masked=Y-X_masked@new_lincoeff_masked
                rothe_error_gaussian_removed[i-len(self.params_frozen)//4]+=np.linalg.norm(zs_masked)**2
            for error in rothe_error_gaussian_removed:
                print(error)
        print(rothe_error)
        #self.optimal_lincoeff=new_lincoeff
        return rothe_error
    def rothe_plus_gradient_logscale(self,nonlin_params_unfrozen):
        error,gradient=self.rothe_plus_gradient(nonlin_params_unfrozen,False)
        log_err=np.log(error)
        log_grad=gradient/error
        return log_err,log_grad
    def orthonormalize_orbitals(self,nonlin_params,old_lincoeff,orbital_norms=None):
        old_action=self.old_action *sqrt_weights
        functions,minus_half_laplacians=setupfunctions(nonlin_params.reshape((-1,4)),points)
        functions=functions.T
        functions2= functions* sqrt_weights.reshape(-1, 1)
        ovlp_matrix=np.conj(functions2.T)@functions2
        if orbital_norms is None:
            orbital_norms=np.ones(old_action.shape[0])
        
        """
        #Orthogonal Procrustes for some reason gives absolutely terrible results???
        D_mat_inv_minhalf=np.diag(1/np.sqrt(orbital_norms))
        D_mat_inv_plushalf=np.diag(np.sqrt(orbital_norms))

        #fock_act_on_new_gauss=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
        #                                            num_gauss=len(functions),time_dependent_potential=self.pot,
        #                                            functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        #X=functions+1j*self.dt/2*fock_act_on_new_gauss
        #X=X.T
        #X = X * sqrt_weights.reshape(-1, 1)

        eigvals,eigvecs=np.linalg.eigh(ovlp_matrix)
        print(eigvals[:3])
        eigvals_sqrt=np.sqrt(eigvals)
        eigvals_sqrt_inv=(1/eigvals_sqrt)
        ovlp_matrix_inv_minhalf=eigvecs@np.diag(eigvals_sqrt_inv)@eigvecs.T.conj()
        decomposition_matrix=D_mat_inv_minhalf@np.conj(old_action)@X@ovlp_matrix_inv_minhalf
        U,sigma,Vdagger=linalg.svd(decomposition_matrix,full_matrices=False)

        c_dagger_tilde=U@Vdagger
        c_dagger=D_mat_inv_plushalf@c_dagger_tilde@ovlp_matrix_inv_minhalf
        c_new=c_dagger.conj().T
        rothe_error = np.linalg.norm(old_action.T - X @ old_lincoeff, ord='fro')**2
        """
        ovlp_matrix_MO_basis = np.conj(old_lincoeff).T @ ovlp_matrix @ (old_lincoeff)
        previous_norm=np.diag(ovlp_matrix_MO_basis)
        # Diagonalize the overlap matrix to get eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(ovlp_matrix_MO_basis)
        S_pow12_inv = np.diag(eigvals**(-0.5))  # S^(-1/2)
        new_lincoeff =(old_lincoeff @ eigvecs @ S_pow12_inv @ eigvecs.T.conj())*np.sqrt(orbital_norms)
        self.optimal_lincoeff=new_lincoeff
        return new_lincoeff
def apply_mask(nonlin_params_old,lincoeff,nbasis,nfrozen):
    new_params=nonlin_params_old.copy()
    orbitals_before_mask=make_orbitals(lincoeff,nonlin_params_old)
    orbitals_masked=cosine_mask*orbitals_before_mask
    norbs=orbitals_before_mask.shape[0]
    nonlin_params_frozen=nonlin_params_old[:4*nfrozen]
    new_lincoeff_optimal=lincoeff.copy()
    def error_and_deriv(nonlin_params_new,return_best_lincoeff=False):
        nonlin_params=np.concatenate((nonlin_params_frozen,nonlin_params_new))
        functions,minus_half_laplacians,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(nonlin_params.reshape((-1,4)),points)
        function_derivs=[]
        for i in range(nfrozen,len(aderiv_funcs)):
            function_derivs+=[aderiv_funcs[i],bderiv_funcs[i],pderiv_funcs[i],qderiv_funcs[i]]
        X=functions
        n_gridpoints=X.shape[1]
        new_lincoeff=np.empty((nbasis,norbs),dtype=np.complex128)
        old_action=orbitals_masked*sqrt_weights
        X=X.T
        Xderc=np.zeros_like(X)
        X = X * sqrt_weights.reshape(-1, 1)
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
            invmats.append(np.linalg.inv(XTX)) #+ lambd * I)) ? the "small c" is already included previously
            new_lincoeff[:,orbital_index]=invmats[-1]@XTy
            zs[orbital_index]=Y-X@new_lincoeff[:,orbital_index]
            rothe_error+=np.linalg.norm(zs[orbital_index])**2
        Xders=np.array(function_derivs)
        Xders=Xders.T
        Xders = Xders * sqrt_weights.reshape(-1, 1)
        Xders=Xders
        gradient=np.zeros_like(nonlin_params_new)
        for i in range(len(nonlin_params_new)):
            Xder=Xderc.copy()
            Xder[:,nfrozen+i//4]=Xders[:,i]
            Xder_dag=Xder.conj().T
            for orbital_index in range(old_action.shape[0]):
                Y=old_action[orbital_index]
                invmat=invmats[orbital_index]
                XTYder=Xder_dag @ Y
                XTY=X_dag @ Y
                matrix_der=-invmat@(X_dag@Xder+Xder_dag@X)@invmat
                cder=matrix_der@XTY+invmat@XTYder
                gradvec=(-Xder@new_lincoeff[:,orbital_index]-X@cder)
                gradient[i]+=2*np.real(zs[orbital_index].conj().T@gradvec)
        if return_best_lincoeff:
            return rothe_error,gradient,new_lincoeff
        return rothe_error,gradient
    nit=0
    initial_mask_error,gradient_mask=error_and_deriv(nonlin_params_old[4*nfrozen:])
    error_due_to_mask=initial_mask_error
    if initial_mask_error>1e-10:
        solution,new_rothe_error,time,nit=minimize_transformed_bonds(error_and_deriv,
                                                        start_params=nonlin_params_old[4*nfrozen:],
                                                        gradient=True,
                                                        maxiter=50,
                                                        gtol=1e-8,
                                                        both=True,
                                                        lambda_grad0=1e-10,
                                                        scale=scale)
        new_params=np.concatenate((nonlin_params_frozen,solution))
        error_due_to_mask,grad,new_lincoeff_optimal=error_and_deriv(solution,True)
        print("Niter: %d, Error due to mask: %.2e/%.2e, time: %.1f"%(nit,error_due_to_mask,initial_mask_error,time))
    if (nit>=1 or error_due_to_mask>1e-11) and grid_b<100:
        print("You should increase the grid size and rerun from a previous time step")
        sys.exit()
    functions,minus_half_laplacians=setupfunctions(new_params.reshape((-1,4)),points)
    functions=functions.T
    functions2= functions* sqrt_weights.reshape(-1, 1)
    ovlp_matrix=np.conj(functions2.T)@functions2

    ovlp_matrix_MO_basis=np.conj(new_lincoeff_optimal).T@ovlp_matrix@new_lincoeff_optimal
    eigvals,eigvecs=np.linalg.eigh(ovlp_matrix_MO_basis)
    new_lincoeff_optimal=new_lincoeff_optimal@eigvecs #Orthogonalize

    return new_params,new_lincoeff_optimal,eigvals

class Rothe_propagation:
    def __init__(self,params_initial,lincoeffs_initial,pulse,timestep,points,nfrozen=0,t=0,norms=None,params_previous=None):
        self.nbasis=lincoeffs_initial.shape[0]
        self.norbs=lincoeffs_initial.shape[1]
        if norms is not None:
            self.norms=norms
        else:
            self.norms=np.ones(self.norbs)
        self.pulse=pulse
        self.dt=timestep
        params_initial=params_initial.flatten()
        self.lincoeffs=lincoeffs_initial
        self.params=params_initial
        self.functions=None
        self.nfrozen=nfrozen
        if params_previous is not None:
            try:
                self.adjustment=params_initial[4*self.nfrozen:]-params_previous[4*self.nfrozen:]
            except:
                self.adjustment=None
        else:
            self.adjustment=None
        self.full_params=np.concatenate((lincoeffs_initial.flatten().real,lincoeffs_initial.flatten().imag,params_initial))
        self.t=t
        self.lambda_grad0=1e-14
    def propagate(self,t,maxiter):
        initial_lincoeffs=self.lincoeffs
        initial_params=self.params
        self.time_dependent_potential=self.pulse(t+self.dt/2)*points
        initial_full_new_params=initial_params[4*self.nfrozen:]
        rothe_evaluator=Rothe_evaluator(initial_params,initial_lincoeffs,self.time_dependent_potential,dt,self.nfrozen)
        initial_rothe_error,grad0=rothe_evaluator.rothe_plus_gradient(initial_full_new_params)
        start_params=initial_params[4*self.nfrozen:]
        best_start_params=start_params.copy()
        ls=np.linspace(0,1,11)
        ls=[0,0.5,0.9,1,1.1]
        best=0
        if self.adjustment is not None:
            updated_res=[initial_rothe_error]
            optimal_linparams=[rothe_evaluator.optimal_lincoeff]
            dx=self.adjustment
            for i in ls[1:]:
                changed=initial_full_new_params+i*dx
                for i in range(len(changed)//4):
                    if changed[i*4]<avals_min:
                        changed[i*4]=avals_min
                    elif changed[i*4]>avals_max:
                        changed[i*4]=avals_max
                    if changed[i*4+1]<bvals_min:
                        changed[i*4+1]=bvals_min
                    elif changed[i*4+1]>bvals_max:
                        changed[i*4+1]=bvals_max
                    
                    if changed[i*4+2]<pvals_min:
                        changed[i*4+2]=pvals_min
                    elif changed[i*4+2]>pvals_max:
                        changed[i*4+2]=pvals_max
                    
                    if changed[i*4+3]<muvals_min:
                        changed[i*4+3]=muvals_min
                    elif changed[i*4+3]>muvals_max:
                        changed[i*4+3]=muvals_max
                    """"""
                updated_re,discard=rothe_evaluator.rothe_plus_gradient(changed)
                updated_res.append(updated_re)
                optimal_linparams.append(rothe_evaluator.optimal_lincoeff)
            best=np.argmin(updated_res)
            start_params=initial_full_new_params+ls[best]*dx
            best_start_params=start_params.copy()
            initial_rothe_error=updated_res[best]
            optimal_linparam=optimal_linparams[best]
            print("Old Rothe error, using change of %.1f: %e"%(ls[best],initial_rothe_error))
        else:
            print("Old Rothe error: %e"%initial_rothe_error)
            optimal_linparam=rothe_evaluator.optimal_lincoeff
        if molecule=="LiH":
            gtol=1e-11;
        elif molecule=="LiH2":
            gtol=1e-11
        if scale=="log":
            optimization_function=rothe_evaluator.rothe_plus_gradient_logscale
            if molecule=="LiH":
                gtol=5e-1
            elif molecule=="LiH2":
                gtol=1e-3
        else:
            optimization_function=rothe_evaluator.rothe_plus_gradient
        if optimize_untransformed:
            hess_inv0=np.diag(1/abs(grad0+self.lambda_grad0*np.array(len(grad0))))
            sol=minimize(optimization_function,
                         start_params,jac=True,
                         method='BFGS',
                         options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':gtol})
            solution=sol.x
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            print("Number of iterations: ",sol.nit)
            new_rothe_error,newgrad=optimization_function(solution)
            print("Rothe Error after optimization: %e using lambd=%.1e"%(new_rothe_error,self.lambda_grad0))
            print(list(sol.x))
        else:
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
                                                        start_params=start_params,
                                                        gradient=True,
                                                        maxiter=maxiter,
                                                        gtol=gtol,
                                                        both=True,
                                                        lambda_grad0=self.lambda_grad0,
                                                        scale=scale)
            #rothe_evaluator.rothe_error_oneremoved(solution)
            #sys.exit(0)
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            if scale=="log":
                new_rothe_error=np.exp(new_rothe_error)
            print("RE after opt: %.2e/%.2e, Ngauss=%d, time=%.1f, niter=%d/%d"%(new_rothe_error,rothe_epsilon_per_timestep**2,len(solution)//4,time,niter,maxiter))
        sqrt_RE=sqrt(new_rothe_error)
        if sqrt_RE>rothe_epsilon_per_timestep:
            print("We have to add more Gaussians, Rothe error is too big")
            self.nbasis+=1
            rothe_evaluator.nbasis=self.nbasis
            new_params_list=[]
            rothe_errors=[]
            avals=solution[0::4]
            bvals=solution[1::4]
            pvals=solution[2::4]
            muvals=solution[3::4]
            minvals=np.array([avals_min,bvals_min,pvals_min,muvals_min])
            maxvals=np.array([avals_max,bvals_max,pvals_max,muvals_max])
            multipliers_min=np.array([1.05,0.95,0.95,0.95])
            number_of_randoms=400
            avals_sample=get_Guess_distribution(avals,number_of_randoms)
            bvals_sample=get_Guess_distribution(bvals,number_of_randoms)
            pvals_sample=get_Guess_distribution(pvals,number_of_randoms)
            muvals_sample=get_Guess_distribution(muvals,number_of_randoms)
            for k in range(number_of_randoms):
                random_params=[avals_sample[k],bvals_sample[k],pvals_sample[k],muvals_sample[k]]
                for i in range(4):
                    if random_params[i]<minvals[i]:
                        random_params[i]=minvals[i]*multipliers_min[i]
                    elif random_params[i]>maxvals[i]:
                        random_params[i]=maxvals[i]*0.95
                new_params=np.concatenate((solution,random_params))
                new_rothe_error,grad=rothe_evaluator.rothe_plus_gradient(new_params)
                new_params_list.append(new_params)
                rothe_errors.append(new_rothe_error)
            best=np.argmin(rothe_errors)
            print("Rothe error before optimization: %e"%rothe_errors[best])  
            best_new_params=new_params_list[best]
            solution_temp=best_new_params
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
                                                        start_params=solution_temp,
                                                        gradient=True,
                                                        maxiter=500,
                                                        gtol=gtol*1e-3,
                                                        both=True,
                                                        multi_bonds=1,
                                                        lambda_grad0=self.lambda_grad0,
                                                        scale=scale,
                                                        intervene=False)
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            print("Rothe error after optimization: %e"%new_rothe_error)
        self.last_rothe_error=sqrt_RE
        new_params=np.concatenate((initial_params[:4*self.nfrozen],solution))
        

        
        #After opatimization: Make sure orbitals are orthonormal, and apply mask
        new_lincoeff=rothe_evaluator.orthonormalize_orbitals(new_params,new_lincoeff,self.norms)
        new_params,new_lincoeff,self.norms=apply_mask(new_params,new_lincoeff,self.nbasis,self.nfrozen)
        #Reorthogonalize the orbitals, but nor reorthonormalize
        new_lincoeff=rothe_evaluator.orthonormalize_orbitals(new_params,new_lincoeff,self.norms)

        C_flat=new_lincoeff.flatten()
        linparams_new=np.concatenate((C_flat.real,C_flat.imag))
        self.full_params=np.concatenate((linparams_new,initial_params[:4*self.nfrozen],solution))
        try:
            self.adjustment=solution-initial_full_new_params
        except ValueError:
            len_init=len(initial_full_new_params)
            self.adjustment=solution[:len_init]-initial_full_new_params
            self.adjustment=np.concatenate((self.adjustment,np.zeros(len(solution)-len_init)))
        self.params=new_params
        self.lincoeffs=new_lincoeff
        avals=self.params[4*self.nfrozen::4]
        print("Avals min and max: %.3e, %.3e; Norms"%(np.min(avals),np.max(avals)),self.norms)
    def propagate_nsteps(self,Tmax,maxiter):
        filename="Rothe_wavefunctions_%s_%.4f_%d_%d_%d_%.3e.npz"%(molecule,E0,initlen,num_gauss,maxiter,rothe_epsilon)
        if self.t==0:
            #Delete the file fith filename if it exists
            try:
                os.remove(filename)
            except:
                pass
            x_expectation=calculate_x_expectation(self.lincoeffs,self.params)
            if molecule=="LiH":
                initnorms=[1.0,1.0]
            elif molecule=="LiH2":
                initnorms=[1.0,1.0,1.0,1.0]
            save_wave_function(filename, self.full_params, self.dt,x_expectation,self.t,0,initnorms,self.nbasis,self.norbs)
        while self.t<Tmax:
            self.propagate(self.t,maxiter)
            self.t+=self.dt
            #F,S=calculate_Fock_and_overlap(self.lincoeffs,self.params,time_dependent_potential=self.time_dependent_potential)
            x_expectation=calculate_x_expectation(self.lincoeffs,self.params)
            save_wave_function(filename, self.full_params, self.dt,x_expectation,self.t,self.last_rothe_error,self.norms,self.nbasis,self.norbs)
            
    def plot_orbitals(self,t):
        plt.figure()
        orbitals=make_orbitals(self.lincoeffs,self.params)
        for i in range(self.norbs):
            plt.plot(points,np.abs(orbitals[i]),label="|Orbital %d|"%i)
        plt.legend()
        plt.savefig("Oribtals_t=%.4f.png"%(t), dpi=200)
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

initlen=int(sys.argv[1])
n_extra=int(sys.argv[2])
F_input = float(sys.argv[3])# Maximum field strength in 10^14 W/cm^2
maxiter=int(sys.argv[4])
start_time=float(sys.argv[5])
molecule=sys.argv[6]
freeze_start=sys.argv[7]
scale=sys.argv[8]
try:
    optimize_untransformed=bool(sys.argv[10])
except:
    optimize_untransformed=False
try:
    rothe_epsilon=float(sys.argv[9])
except:
    rothe_epsilon=100
if freeze_start=="freeze":
    nfrozen=initlen
else:
    nfrozen=0

inner_grid=17
if F_input==1:
    grid_b=150
elif F_input==4:
    grid_b=150
elif F_input==8:
    grid_b=600
grid_a=-grid_b
muvals_max=grid_b-10
muvals_min=grid_a+10

points_inner,weights_inner=gaussian_quadrature(-inner_grid,inner_grid,14*inner_grid+1)
points_outer1,weights_outer1=trapezoidal_quadrature(grid_a, -inner_grid, int(2.5*(grid_b-inner_grid)))
points_outer2,weights_outer2=trapezoidal_quadrature(inner_grid, grid_b, int(2.5*(grid_b-inner_grid)))
points=np.concatenate((points_outer1,points_inner,points_outer2))
weights=np.concatenate((weights_outer1,weights_inner,weights_outer2))
n=len(points)
lambd=1e-9 #Should be at most 1e-8, otherwise the <x(t)> will become wrongly oscillating
cosine_mask=cosine4_mask(points,grid_a+5,grid_b-5)
sqrt_weights=np.sqrt(weights)
if molecule=="LiH":
    R_list=[-1.15, 1.15]
    Z_list=[3,1]
elif molecule=="LiH2":
    R_list=[-4.05, -1.75, 1.75, 4.05]
    Z_list=[3, 1,3,1]
norbs=sum(Z_list)//2
alpha=0.5

V=external_potential=calculate_potential(Z_list,R_list,alpha,points)
wT=weights.T
e_e_grid=e_e_interaction(points)
weighted_e_e_grid = e_e_grid * weights[:, np.newaxis]

if initlen==10 and molecule=="LiH":
    gaussian_nonlincoeffs=[np.float64(0.8005548344751579), np.float64(0.38388098086759453), np.float64(0.05011237373133904), np.float64(-1.5706492999278023), np.float64(0.5471791250933454), np.float64(0.1496760887228818), np.float64(0.024834786050146974), np.float64(-0.5626307864355159), np.float64(1.642958175075867), np.float64(0.2578356129265224), np.float64(-0.0826351235103536), np.float64(-1.166939922382098), np.float64(0.6237626939764296), np.float64(-0.2769813412742149), np.float64(2.0288327736859637), np.float64(-2.3902521361804943), np.float64(0.5183977106271925), np.float64(-0.10734666957309137), np.float64(-1.0395471674448307), np.float64(0.35371966444219743), np.float64(1.0791519263254514), np.float64(0.5619527442898424), np.float64(0.5202776863119151), np.float64(-0.18375219928922232), np.float64(0.2837535242449411), np.float64(0.019464029966476802), np.float64(0.20661082283950774), np.float64(-0.09685727064199649), np.float64(0.45688824689247265), np.float64(0.3274078868425557), np.float64(-2.730789997946961), np.float64(-2.4443909680924523), np.float64(0.5079702236453066), np.float64(0.8241889399259744), np.float64(2.340184721939206), np.float64(1.7373451028877664), np.float64(0.643322475143318), np.float64(-0.562428250612366), np.float64(2.313139832059642), np.float64(-4.543334041417847)]

elif initlen==20 and molecule=="LiH":
    gaussian_nonlincoeffs=[array([7.91752880e-01, 2.42411512e-02, -8.97476398e-01, -1.05426591e+00]), array([3.44209833e-01, -9.79200779e-03, -4.25725764e-01, -8.89067696e-01]), array([1.10011167e+00, -2.49999130e-01, -3.97568745e-01, -8.52554309e-01]), array([-1.72381478e+00, 3.58680812e-01, -6.35054994e-01, -6.38924163e-01]), array([2.84780973e-01, -1.04409022e-02, -2.40506761e-01, -2.92563322e-01]), array([-4.03130626e-01, -4.16630519e-02, -4.19315963e-01, 2.29472896e-01]), array([4.10671105e-01, 6.76167290e-03, -1.22117270e-02, 3.08399240e-01]), array([-2.11109542e+00, 2.60207725e+00, 5.41978256e-01, -5.08578423e-01]), array([5.56994436e-01, 3.31417482e-03, 5.43611001e-02, 2.03213569e-01]), array([-2.17670221e-01, -3.04621253e-03, -1.35643646e-01, -2.45577212e-01]), array([1.79521641e+00, -1.54035060e+00, -1.67717421e+00, -2.46080442e-01]), array([-6.73953961e-01, 6.42508815e-01, -3.67737731e+00, -2.55015891e+00]), array([1.07193706e+00, -1.31211950e+00, -1.15736230e+00, 6.45474797e-01]), array([1.32912630e-01, -1.41993209e-01, -3.54309883e-02, 1.58943047e+00]), array([-1.77463854e+00, 1.35236137e+00, -1.76753088e-01, 3.25592878e-01]), array([-7.83710486e-01, -6.67898734e-01, -7.36336834e-01, 2.26507329e-02]), array([-6.83487053e-01, -2.52273486e-01, -1.33419578e+00, -3.48094881e-01]), array([1.61730999e+00, -2.02595956e+00, -1.64463690e+00, 1.21053317e-01]), array([-7.94485475e-01, -6.30128349e-01, -1.67003985e+00, 6.21992624e-01]), array([-8.90181229e-01, -1.00833765e+00, -2.04045879e-02, 3.02777697e-01])]
    gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs).flatten()
elif initlen==22 and molecule=="LiH":
    gaussian_nonlincoeffs=[np.float64(1.3755383332751012), np.float64(0.43387999324534177), np.float64(-0.4858312408142174), np.float64(-0.8999872210482782), np.float64(1.2175683170252989), np.float64(0.6269086499992571), np.float64(-1.1297585667797985), np.float64(-1.3083638837633915), np.float64(2.5836785486576272), np.float64(-0.13908524641274367), np.float64(0.1618322906006488), np.float64(-1.1556832556324297), np.float64(1.4707404911635453), np.float64(-2.0202154822674716), np.float64(0.572145181961057), np.float64(-0.5391821122667797), np.float64(0.9567643536857007), np.float64(-0.7463513960014231), np.float64(-2.443752067237098), np.float64(0.2875372105851867), np.float64(1.7591398989492861), np.float64(2.483696383276731), np.float64(1.6284324792658145), np.float64(0.12781175529768932), np.float64(0.6390810554679798), np.float64(0.049263101591370456), np.float64(2.0737171999977715), np.float64(-0.30089959455706317), np.float64(2.044014866004284), np.float64(-1.3427709212620054), np.float64(0.7379085358667082), np.float64(-1.857737410997086), np.float64(1.2603037750608468), np.float64(1.9023934363841728), np.float64(3.2837577074801647), np.float64(0.7409696237648654), np.float64(1.1383755749450208), np.float64(1.3180932314577356), np.float64(2.689743407303616), np.float64(0.6121194764401825), np.float64(0.453146950645606), np.float64(0.043132340856857965), np.float64(0.2914110279958246), np.float64(0.6313473515069905), np.float64(0.7726193127834898), np.float64(-0.07094643262477693), np.float64(-0.2628469396156622), np.float64(0.6540779764769042), np.float64(0.5671626451904503), np.float64(0.011194824379320498), np.float64(-0.11062701278825922), np.float64(-0.7596938288138286), np.float64(0.8882971786225393), np.float64(0.37040408863222857), np.float64(-0.7489040046593811), np.float64(-1.1556924726848299), np.float64(0.6971761909160318), np.float64(-0.2020207048820606), np.float64(1.2604114638965187), np.float64(-1.712728763041013), np.float64(0.33835903075069357), np.float64(-0.038410665716412565), np.float64(-0.3469750826941411), np.float64(0.5797619195158886), np.float64(0.46284142932791744), np.float64(-0.08480751464325252), np.float64(-0.7087556675487822), np.float64(0.6762628664665942), np.float64(0.46634092792162196), np.float64(0.042821166545434976), np.float64(-0.539731155788753), np.float64(-1.7500709382198745), np.float64(0.2448678668042601), np.float64(0.02046331000401414), np.float64(0.3004700412659324), np.float64(0.8419060033541952), np.float64(0.26163426101560894), np.float64(-0.005701242327813436), np.float64(0.19853461777841735), np.float64(-0.7404032386256065), np.float64(0.3801036124488238), np.float64(0.16634071265689343), np.float64(-1.7024811840186083), np.float64(-3.320079350073238), np.float64(0.1888711154067108), np.float64(0.008367892346367154), np.float64(-0.10439053087284329), np.float64(-0.168435496776038)]

elif initlen==25 and molecule=="LiH2":
    gaussian_nonlincoeffs=[np.float64(-1.9371167812031795), np.float64(1.7290585163492986), np.float64(-0.7679741986650402), np.float64(-3.761894355016483), np.float64(1.9454144410927947), np.float64(0.16786317742454546), np.float64(-0.06449013268030306), np.float64(1.7487688941347448), np.float64(0.6273060281080144), np.float64(0.09583370617601904), np.float64(0.03957694358471211), np.float64(-2.2965009569417845), np.float64(0.5461300037711249), np.float64(-0.11619679930284742), np.float64(-0.6618539738704318), np.float64(2.15736431839032), np.float64(0.8556667402576349), np.float64(0.2574519976214483), np.float64(0.9090848508253586), np.float64(2.677115068519693), np.float64(1.3499306392743122), np.float64(0.629096534510376), np.float64(-2.824961481320992), np.float64(-3.508062023372533), np.float64(1.3340372084787528), np.float64(-0.36238866696475913), np.float64(1.4471210300601134), np.float64(2.3402092435529105), np.float64(1.3604835488035285), np.float64(1.076218568973415), np.float64(-1.893124316457211), np.float64(-3.9318655644085396), np.float64(0.2925660546874304), np.float64(-0.03600548517367306), np.float64(-0.5206512424961109), np.float64(1.8356709252082475), np.float64(1.2459722695739355), np.float64(0.917308512243142), np.float64(-1.523746460829836), np.float64(-3.7657488357098625), np.float64(1.4122245182083082), np.float64(-1.2028685436471978), np.float64(-1.4043642536661227), np.float64(-2.5620766032285154), np.float64(0.6325790594519487), np.float64(0.25288764039508743), np.float64(0.276546616475345), np.float64(-2.3939729316779728), np.float64(1.108634931754126), np.float64(-0.43108482142551363), np.float64(0.7186916978741584), np.float64(1.9802759201396725), np.float64(0.7154390113223642), np.float64(0.5142595843148614), np.float64(-1.492942150693892), np.float64(0.781597041995784), np.float64(1.3983442683773213), np.float64(0.9344877669396385), np.float64(0.6526133463758973), np.float64(2.9126735163575157), np.float64(0.5007024423078054), np.float64(0.13914245094339342), np.float64(-0.7837300754921872), np.float64(-3.8643061431941295), np.float64(0.710705850983338), np.float64(0.04053797162029212), np.float64(0.08613494919242075), np.float64(1.9571252133818038), np.float64(0.9732821451743155), np.float64(0.21445401423115346), np.float64(-0.34584944096040326), np.float64(-4.06195418284658), np.float64(0.3053483521918622), np.float64(0.04127395094764254), np.float64(-0.4400946455816779), np.float64(-2.4685406098794243), np.float64(1.3852324153089215), np.float64(-0.2997889392483762), np.float64(1.2023939511427457), np.float64(2.483255919965986), np.float64(0.23383835833600955), np.float64(-0.05036015093037508), np.float64(-1.3261969912726912), np.float64(2.9387102674300576), np.float64(0.34320051561974807), np.float64(0.07529881315833485), np.float64(0.6850022451704646), np.float64(2.518840269328582), np.float64(0.21351517906681386), np.float64(-0.0026735341685827658), np.float64(0.11695717612378112), np.float64(-1.606930376481328), np.float64(0.5142989902200078), np.float64(-0.058009418759063416), np.float64(0.6403380184762146), np.float64(-3.7727461220103815), np.float64(0.21605247124846702), np.float64(-0.02768908517014951), np.float64(-0.590262222795961), np.float64(3.0653534133762936)]

elif initlen==32 and molecule=="LiH2":
    gaussian_nonlincoeffs=[np.float64(-2.2317056359467506), np.float64(-1.2272375095162027), np.float64(0.23360866018413173), np.float64(-3.621824808647527), np.float64(2.403216328962662), np.float64(-0.09979450013943086), np.float64(-0.13783182481844644), np.float64(1.762555668949134), np.float64(1.2474771686136328), np.float64(-0.4394564442939505), np.float64(-1.6921049595367084), np.float64(-2.712598893728889), np.float64(1.483218981973524), np.float64(0.6054130495832375), np.float64(0.29626060865634013), np.float64(2.2108684879175526), np.float64(1.1829047765163097), np.float64(0.9246537041173427), np.float64(-1.4391221807575527), np.float64(0.6483967694153571), np.float64(0.6981437514424434), np.float64(0.12346255358242765), np.float64(-0.21483134486573635), np.float64(-0.365382073239989), np.float64(2.0670902968648526), np.float64(0.031732038310215704), np.float64(0.03890237490585383), np.float64(4.031435744684241), np.float64(1.4574033542977907), np.float64(-1.5106506321870432), np.float64(-0.255524773125416), np.float64(-3.0417807381674145), np.float64(0.8768522198008559), np.float64(0.06815447452859076), np.float64(-0.10011714100768827), np.float64(2.064568863739714), np.float64(1.104936148652444), np.float64(2.034048157506189), np.float64(-2.0700857029721003), np.float64(0.028994190490856133), np.float64(2.0684061672102074), np.float64(0.6382706299827365), np.float64(0.6520573741167575), np.float64(-3.3544155752581544), np.float64(1.1177756233626819), np.float64(0.593713628596617), np.float64(-0.10495786514045681), np.float64(2.873144906275408), np.float64(1.0231319635611482), np.float64(0.2962041484272756), np.float64(-1.5443099406758425), np.float64(-3.0777783985589564), np.float64(1.4170435606738634), np.float64(1.501483554162789), np.float64(1.2688027698569055), np.float64(-2.6279518652953215), np.float64(0.7640614351398998), np.float64(0.3789298559502068), np.float64(-1.7016368248601608), np.float64(1.6885014522816961), np.float64(1.483693070142052), np.float64(0.1644057832543084), np.float64(-0.7393524053834604), np.float64(-3.280301718184856), np.float64(0.53736392873247), np.float64(0.1893715981384544), np.float64(-1.5377129678775954), np.float64(-2.909810577504364), np.float64(0.4773699764726102), np.float64(-0.15829994617389742), np.float64(-0.9736323624870706), np.float64(3.2168447435355967), np.float64(1.0746698359422258), np.float64(-1.2317901490182148), np.float64(-1.2439026785556986), np.float64(2.7651046893760074), np.float64(0.7509177204890064), np.float64(-0.016669384399343404), np.float64(0.05713744177903095), np.float64(-3.051265214254328), np.float64(0.4161613566391621), np.float64(0.024983744445208796), np.float64(-0.2723504883729735), np.float64(1.5028460580049434), np.float64(0.9328273434766154), np.float64(0.15648615221389434), np.float64(-0.6214232507118678), np.float64(-1.7625670779609652), np.float64(0.3952229394804105), np.float64(0.07127438161651693), np.float64(0.5728686554791315), np.float64(3.3629546187417083), np.float64(0.7734998367490572), np.float64(0.13030645305350708), np.float64(1.7664202528224402), np.float64(3.0151619943460406), np.float64(0.24300439366792995), np.float64(-0.02303564156897768), np.float64(0.5116216693989191), np.float64(-2.3976440709459474), np.float64(0.3784152777629207), np.float64(-0.029690195313388887), np.float64(-0.3853845105388067), np.float64(3.1970434579292086), np.float64(0.4474703698655672), np.float64(0.1899872964797411), np.float64(-1.5599523448468515), np.float64(-3.3267482095304772), np.float64(0.5806665174151137), np.float64(0.08237601789060621), np.float64(-0.5968538973338998), np.float64(-3.3162693524593294), np.float64(0.23774440770027833), np.float64(0.02255945106824925), np.float64(-0.31363181896753106), np.float64(-2.4014216258774974), np.float64(0.2633782693076539), np.float64(0.014379738405316181), np.float64(0.2778907927268805), np.float64(2.821450838903747), np.float64(0.15485486974267088), np.float64(0.031162919627486147), np.float64(-0.6328556414039005), np.float64(-2.5500552548012414), np.float64(0.1999723266517023), np.float64(0.026848725753687408), np.float64(0.6886069980836459), np.float64(3.632780830214886)]

elif initlen==34 and molecule=="LiH2":
    gaussian_nonlincoeffs=[np.float64(-2.3919262524131053), np.float64(1.6983680333388047), np.float64(-1.0507669273674207), np.float64(-3.511124618447428), np.float64(2.2246891441959535), np.float64(-0.13089053602504794), np.float64(-0.17296237938305664), np.float64(1.8302203861355923), np.float64(0.7354736387377339), np.float64(-0.3362670686665967), np.float64(-0.4614333746460378), np.float64(-2.0839244749400634), np.float64(0.9866765824901731), np.float64(-0.3006720725711162), np.float64(-0.6505026907281375), np.float64(2.71034635036849), np.float64(0.1925105420435331), np.float64(0.21258900192970512), np.float64(3.3917280225602524), np.float64(5.87281218737565), np.float64(1.698588799982305), np.float64(-0.8087913496137077), np.float64(-4.4665360245006696), np.float64(-2.936204380487738), np.float64(2.186692740415571), np.float64(-0.30926659891893465), np.float64(-0.8985773550935838), np.float64(2.362082210407118), np.float64(2.1503457008631575), np.float64(0.7739469300364852), np.float64(-1.8306157493514539), np.float64(-3.5152716307386296), np.float64(0.5408784376166097), np.float64(-0.08852048131804374), np.float64(-0.43865116695688616), np.float64(2.9435691933917956), np.float64(1.4096402894319138), np.float64(0.5789202909334807), np.float64(-4.451818403330961), np.float64(-3.3843925767540424), np.float64(2.0660672964944933), np.float64(-1.2694854076627586), np.float64(-1.2455129725508018), np.float64(-2.3530945497996267), np.float64(1.2637806124124469), np.float64(-0.36031964996923876), np.float64(-0.18816414866925582), np.float64(-1.8081331406310988), np.float64(1.896060707104642), np.float64(1.4208586460713963), np.float64(-1.7340762691018283), np.float64(1.6366355027322514), np.float64(0.8903848775620167), np.float64(0.6560496288428984), np.float64(-2.2404299798360294), np.float64(1.6363557486890286), np.float64(1.544033003122892), np.float64(-1.649420387112326), np.float64(-0.8044447000249991), np.float64(2.9557757569206284), np.float64(1.1300014328237504), np.float64(-0.2679725006745771), np.float64(0.32658814288195154), np.float64(-3.308372778744816), np.float64(0.9472259540618642), np.float64(-0.4882382263898718), np.float64(1.1373208155118533), np.float64(1.2269024517512013), np.float64(1.5673374416851686), np.float64(1.1301430807385107), np.float64(-0.8991125451785721), np.float64(-3.216276222703561), np.float64(0.7910598224960512), np.float64(-0.18783833608181202), np.float64(0.7904276517213784), np.float64(-3.345772507358129), np.float64(0.6767456750654198), np.float64(-0.7484720694621881), np.float64(1.2823741730114866), np.float64(0.541081102212063), np.float64(1.3088869201179958), np.float64(1.1272058958134208), np.float64(-0.7193969826156209), np.float64(2.630363558560648), np.float64(0.715893836412666), np.float64(-0.0028138590154601815), np.float64(-0.023296596892547223), np.float64(1.8842947905919267), np.float64(0.2932401620987503), np.float64(-0.027722655652250117), np.float64(0.6872651953665556), np.float64(-2.137237644946229), np.float64(0.8889240732020098), np.float64(0.2299689564226617), np.float64(1.5179211433620667), np.float64(-2.3471715853683155), np.float64(0.3121261712700929), np.float64(-0.044672935076271285), np.float64(-0.2968678016304338), np.float64(2.7799289215344225), np.float64(0.44212481917220575), np.float64(-0.06626882525123458), np.float64(0.7832476078145153), np.float64(-3.4169172558834386), np.float64(0.49064978148462823), np.float64(-0.2550952162748644), np.float64(-1.223330690017718), np.float64(4.393403022870566), np.float64(0.19423625449646348), np.float64(-0.002019953168325981), np.float64(-0.09748305446741101), np.float64(2.0945221064782245), np.float64(0.22440510152310666), np.float64(0.00039747976109094814), np.float64(0.24343234979808903), np.float64(-2.00110981707022), np.float64(0.6248912386985435), np.float64(0.0846507923931561), np.float64(-0.2347871521206139), np.float64(-3.667553737401025), np.float64(0.39333178981485367), np.float64(0.139956415598448), np.float64(0.7449565769913732), np.float64(-0.9078523438176727), np.float64(0.2249240876771477), np.float64(0.016598646443333837), np.float64(-0.09059326632898644), np.float64(-2.6550538050936776), np.float64(0.18118439776415118), np.float64(0.10195507391647446), np.float64(1.9526335568555728), np.float64(5.761559498457204), np.float64(0.1997859197467974), np.float64(0.02716188167409728), np.float64(0.16420339762419559), np.float64(2.1096521127267454)]

else:
    raise ValueError("initlen not supported")
gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs)
gaussian_nonlincoeffs[::4]=abs(gaussian_nonlincoeffs[::4])

gaussian_nonlincoeffs=list(np.array(gaussian_nonlincoeffs).reshape(-1,4))
print(initlen,n_extra)
if molecule=="LiH":
    a=5
    b=15
elif molecule=="LiH2":
    a=13
    b=25
nextra_half=n_extra//2
pos_list=np.linspace(a,a+2*(nextra_half-1),nextra_half)
pos_list=np.concatenate((-pos_list,pos_list))
for k in range(len(pos_list)):
    params=[1/sqrt(2),0,0,pos_list[k]]
    gaussian_nonlincoeffs.append(params)

gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs)
print(gaussian_nonlincoeffs.shape)
num_gauss=gaussian_nonlincoeffs.shape[0]
print(num_gauss)
potential_grid=calculate_potential(Z_list,R_list,alpha,points)


onebody_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)
overlap_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)

time_dependent_potential=0.1*points #I. e. 0.1*x - very strong field
run_HF=True if start_time<0.05 else False
if run_HF:
    E,lincoeff_initial,epsilon=calculate_energy(gaussian_nonlincoeffs,return_all=True)
    x_expectation_t0=calculate_x_expectation(lincoeff_initial,gaussian_nonlincoeffs)

F0 =np.sqrt(F_input/(3.50944758*1e2))  # Maximum field strength

E0 = F0  # Maximum field strength
print("Field strength in atomic units: %.4f"%E0)
omega = 0.06075  # Laser frequency
t_c = 2 * np.pi / omega  # Optical cycle
n_cycles = 3
dt=0.05
td = n_cycles * t_c  # Duration of the laser pulse
tfinal = td  # Total time of the simulation
nsteps=int(tfinal/dt)
rothe_epsilon_per_timestep=rothe_epsilon/nsteps
print(tfinal)
t=np.linspace(0,tfinal,1000)
fieldfunc=laserfield(E0, omega, td)

functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs.reshape((num_gauss,4)),points)

Tmax=tfinal
filename="Rothe_wavefunctions_%s_%.4f_%d_%d_%d_%.3e.npz"%(molecule,E0,initlen,num_gauss,maxiter,rothe_epsilon)
try:
    np.load(filename)
    times=np.load(filename)["times"]
    if start_time==0:
        os.remove(filename)
        raise FileNotFoundError
    closest_index = np.abs(times - start_time).argmin()

    tmax=times[closest_index]
    params=np.load(filename)["params"]
    xvals=np.load(filename)["xvals"]
    time_step=np.load(filename)["time_step"]
    rothe_errors=np.load(filename)["rothe_errors"]
    norms=np.load(filename)["norms"]
    nbasis=np.load(filename)["nbasis"]
    np.savez(filename, params=params[:closest_index+1], time_step=time_step,times=times[:closest_index+1],
             xvals=xvals[:closest_index+1],rothe_errors=rothe_errors[:closest_index+1],
             norms=norms[:closest_index+1],nbasis=nbasis[:closest_index+1])
    ngauss=nbasis[closest_index]
    norms_initial=norms[closest_index]
    ngauss_wrong=len(params[closest_index])//(4+norbs*2)
    lincoeff_initial_real=params[closest_index][:ngauss*norbs]#.reshape((ngauss,norbs))
    lincoeff_initial_complex=params[closest_index][ngauss_wrong*norbs:(ngauss+ngauss_wrong)*norbs]#.reshape((ngauss,norbs))
    lincoeff_initial=lincoeff_initial_real+1j*lincoeff_initial_complex
    lincoeff_initial=lincoeff_initial.reshape((ngauss,norbs))
    if ngauss_wrong-ngauss>0:
        gaussian_nonlincoeffs=params[closest_index][ngauss_wrong*norbs*2:-4*(ngauss_wrong-ngauss)]
        try:
            gaussian_nonlincoeffs_prev=params[closest_index-1][ngauss_wrong*norbs*2:-4*(ngauss_wrong-ngauss)]
        except:
            gaussian_nonlincoeffs_prev=None
    else:
        gaussian_nonlincoeffs=params[closest_index][ngauss*norbs*2:]
        try:
            gaussian_nonlincoeffs_prev=params[closest_index-1][ngauss*norbs*2:]
        except:
            gaussian_nonlincoeffs_prev=None
except FileNotFoundError:
    tmax=0
    norms_initial=np.ones(norbs)
    gaussian_nonlincoeffs_prev=None
rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,
                                timestep=dt,points=points,nfrozen=nfrozen,t=tmax,norms=norms_initial,params_previous=gaussian_nonlincoeffs_prev)

rothepropagator.propagate_nsteps(Tmax,maxiter)
