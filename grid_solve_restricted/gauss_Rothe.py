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
np.set_printoptions(linewidth=300, precision=5, suppress=True, formatter={'float': '{:0.8e}'.format})
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
            if error_list[i]/error_list[i-i_min]>0.995:
                break
    #sys.exit(0)
    print("Niter: %d"%len(error_list))
    best_error=np.argmin(error_list)
    return param_list[best_error],error_list[best_error]
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
    mmins[0::4]=0.03
    mmaxs[0::4]=10
    mmins[1::4]=-30
    mmaxs[1::4]=30
    mmins[2::4]=-70
    mmaxs[2::4]=70
    mmaxs[3::4]=grid_b
    mmins[3::4]=grid_a
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
    hess_inv0=np.diag(1/abs(grad0+1e-16*np.array(len(grad0))))
    if both is False:
        sol=minimize(transformed_error,transformed_params,jac=transformed_gradient,method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':1e-9})
    else:
        sol=minimize(transform_error_and_gradient,transformed_params,jac=True,method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':1e-9})
    
    transformed_sol=sol.x
    end=time.time()
    print("  REG: Time to optimize: %.4f seconds, niter : %d"%(end-start,sol.nit))
    return untransform_params(transformed_sol), sol.fun

n=int(1500)
grid_a, grid_b = -210, 210
lambd=1e-8 #Should be at most 1e-8, otherwise the <x(t)> will become wrongly oscillating
#points, weights = tanh_sinh_quadrature(grid_a, grid_b, n)
points, weights = trapezoidal_quadrature(grid_a, grid_b, n)
#points,weights=gaussian_quadrature(grid_a, grid_b, n)
cosine_mask=cosine4_mask(points,grid_a+10,grid_b-10)
plt.plot(points,cosine_mask)
plt.show()
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
    gaussian_nonlincoeffs=[array([1.62852040e+00, -1.99085421e+00, -1.28070443e+00, 2.75843219e-01]), array([3.37784970e-01, -2.34298030e-01, 8.96226542e-01, -5.52777247e-01]), array([2.24711248e+00, -3.45570715e-01, -2.62632646e-01, -1.19300927e+00]), array([5.92336566e-01, -6.19077383e-01, 8.21685194e-01, -2.67469331e-01]), array([1.52759963e+00, 6.43956526e-01, -1.52164785e+00, -8.04475147e-01]), array([5.48636134e-01, -2.52766365e-01, -9.03932804e-01, -1.91847153e-01]), array([3.54919119e-01, -5.80327741e-02, -4.33751898e-01, -3.24336393e-01]), array([-5.17908922e-01, 8.55155573e-02, 8.16647835e-01, 4.16672509e-01]), array([9.29852797e-01, -1.50717978e+00, -2.21150149e+00, 1.21004135e+00]), array([3.64646131e-01, -3.46218261e-01, 8.92035085e-02, 2.53860991e-01]), array([8.33235167e-01, 8.46731052e-01, -2.71802098e+00, -1.68544090e+00]), array([4.48036489e-01, -1.81494288e-01, 5.21196555e-01, -9.88132815e-01]), array([1.33103218e+00, -6.79031468e-01, -1.54764226e+00, -4.51143414e-02]), array([5.40735929e-01, -5.59620527e-01, 8.94127104e-01, -7.08536018e-01]), array([1.35240136e+00, 6.29765477e-01, -6.56158551e-01, -5.10202779e-01]), array([3.66225581e-01, -1.23996836e-01, -5.51452037e-01, 7.81454108e-02]), array([3.66126796e-01, -4.70428052e-02, 8.19945927e-02, 9.50195335e-02]), array([-4.82328993e-01, 3.41261099e-02, 1.92353707e+00, 8.08396453e-01]), array([2.42120797e-01, -6.35882995e-03, 4.22265724e-01, 4.32073584e-01]), array([-2.77733361e-01, -1.45592750e-02, 3.16684171e-01, 1.68298047e-01])]


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
    if n_extra>=6:
        pos_list=np.concatenate([np.linspace(-15,-5,n_extra//2),np.linspace(5,15,n_extra//2)])
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

#@jit(nopython=True,fastmath=True,cache=False)
def calculate_Fgauss_fast(fockOrbitals,num_gauss,time_dependent_potential,functions,minus_half_laplacians):
    nFock=len(fockOrbitals)
    #num_gauss=len(functions)
    Fgauss=np.zeros_like(functions)
    Fgauss+=minus_half_laplacians
    if time_dependent_potential is not None:
        potential_term = potential_grid + time_dependent_potential
    else:
        potential_term = potential_grid
    Fgauss+=potential_term*functions
    coulomb_terms=np.zeros((nFock,fockOrbitals.shape[1]),dtype=np.complex128)
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
    def rothe_plus_gradient(self,nonlin_params_unfrozen,n_gauss,hessian=False):
        old_action=self.old_action *sqrt_weights
        gradient=np.zeros_like(nonlin_params_unfrozen)
        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions,minus_half_laplacians,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(nonlin_params.reshape((-1,4)),points)
        fock_act_on_new_gauss=calculate_Fgauss_fast(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions),time_dependent_potential=self.pot,
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
        n_gridpoints=X.shape[1]
        n_params=len(nonlin_params_unfrozen)
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
                                                    num_gauss=len(function_derivs),time_dependent_potential=self.pot,
                                                    functions=np.array(function_derivs),minus_half_laplacians=np.array(kin_derivs))
        
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
    def calculate_(self,nonlin_params_unfrozen,n_gauss):
        jac=np.zeros_like(nonlin_params_unfrozen)
        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions,minus_half_laplacians=setupfunctions(nonlin_params.reshape((num_gauss,4)),points)
        self.orbitals_that_represent_Fock=make_orbitals(self.old_lincoeff,self.old_params)
        X=functions
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
    def propagate(self,t):
        initial_lincoeffs=self.lincoeffs
        initial_params=self.params
        self.time_dependent_potential=self.pulse(t+self.dt/2)*points
        initial_full_new_params=initial_params[4*self.nfrozen:]
        rothe_evaluator=Rothe_evaluator(initial_params,initial_lincoeffs,self.time_dependent_potential,dt,self.nfrozen)
        initial_rothe_error,grad0=rothe_evaluator.rothe_plus_gradient(initial_full_new_params,self.nbasis)
        start_params=initial_params[4*self.nfrozen:]
        maxiter=100
        ls=np.linspace(0,1,11)
        best=0
        if self.adjustment is not None:
            updated_res=[initial_rothe_error]
            dx=self.adjustment
            for i in ls[1:]:
                start_params_alt=initial_full_new_params+i*dx

                updated_re,discard=rothe_evaluator.rothe_plus_gradient(initial_full_new_params,self.nbasis)
                updated_res.append(updated_re)
            best=np.argmin(updated_res)
            start_params_alt=initial_full_new_params+best*dx
            print("Old Rothe error, using change of %.1f: %e"%(ls[best],updated_res[best]))
            start_params=start_params_alt
        else:
            print("Old Rothe error: %e"%initial_rothe_error)
        optimize_untransformed=False
        if optimize_untransformed:
            hess_inv0=np.diag(1/abs(grad0+1e-16*np.array(len(grad0))))
            sol=minimize(rothe_evaluator.rothe_plus_gradient,
                         start_params,jac=True,args=(self.nbasis),
                         method='BFGS',
                         options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':1e-9})
            solution=sol.x
            print("Number of iterations: ",sol.nit)
            new_rothe_error,newgrad=rothe_evaluator.rothe_plus_gradient(solution,self.nbasis)
        else:
            solution,new_rothe_error=minimize_transformed_bonds(rothe_evaluator.rothe_plus_gradient,
                                                        start_params=start_params,
                                                        gradient=True,
                                                        num_gauss=len(initial_lincoeffs),
                                                        num_frozen=self.nfrozen,
                                                        num_orbitals=2,
                                                        multi_bonds=0,
                                                        maxiter=maxiter,
                                                        both=True)
        #solution,new_rothe_error=minimize_hessian(rothe_evaluator.rothe_plus_gradient,start_params,len(initial_lincoeffs))
        print("New Rothe error: %e"%new_rothe_error)
        C_flat=rothe_evaluator.optimal_lincoeff.flatten()
        #C_flat=initial_lincoeffs.flatten()
        linparams_new=np.concatenate((C_flat.real,C_flat.imag))
        self.full_params=concatenated_params=np.concatenate((linparams_new,initial_params[:4*self.nfrozen],solution))
        self.adjustment=solution-start_params
       
        self.params=np.concatenate((initial_params[:4*self.nfrozen],solution))
        self.lincoeffs=rothe_evaluator.optimal_lincoeff
        print(np.reshape(solution,(-1,4)))
    def propagate_nsteps(self,Tmax):
        x_expectations=[]
        times=np.linspace(0,nsteps*self.dt,nsteps+1)
        filename="Rothe_wavefunctions%.4f_%d_%d.npz"%(E0,initlen,num_gauss)
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
            print("Time %.4f, <x>: %.6f"%(self.t,x_expectation))
            #self.plot_orbitals((i+1)*self.dt)
        return x_expectations
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
time_dependent_potential=0.1*points #I. e. 0.1*x - very strong field

E,lincoeff_initial,epsilon=calculate_energy(gaussian_nonlincoeffs,return_all=True)
F_input = float(sys.argv[2])# Maximum field strength in 10^14 W/cm^2
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
print(tfinal)
t=np.linspace(0,tfinal,1000)
fieldfunc=laserfield(E0, omega, td)


x_expectation_t0=calculate_x_expectation(lincoeff_initial,gaussian_nonlincoeffs)
functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs.reshape((num_gauss,4)),points)

Tmax=tfinal
if typer=="linear":
    rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,timestep=dt,points=points,nfrozen=initlen)

    x_expectations=rothepropagator.propagate_nsteps_linear(nsteps)
    x_expectations=[x_expectation_t0]+x_expectations
    plt.plot(np.linspace(0,nsteps*dt,nsteps+1),x_expectations)
    np.savez("Rothe_linearbasis_%.4f_%d.npz"%(E0,num_gauss),t=np.linspace(0,nsteps*dt,nsteps+1),x=x_expectations)
    plt.show()
elif typer=="basis":
    rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,timestep=dt,points=points,nfrozen=initlen)

    x_expectations=rothepropagator.propagate_nsteps_basis(nsteps)
    x_expectations=[x_expectation_t0]+x_expectations
    plt.plot(np.linspace(0,nsteps*dt,nsteps+1),x_expectations)
    np.savez("linearbasis_%.4f_%d.npz"%(E0,num_gauss),t=np.linspace(0,nsteps*dt,nsteps+1),x=x_expectations)
    plt.show()
elif typer=="nonlinear":
    
    filename="Rothe_wavefunctions%.4f_%d_%d.npz"%(E0,initlen,num_gauss)
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
    np.savez("Rothe_%.4f_%d.npz"%(E0,num_gauss),t=np.linspace(0,nsteps*dt,nsteps+1),x=x_expectations)
    plt.show()