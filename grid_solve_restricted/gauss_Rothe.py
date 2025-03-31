import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numba
from numba import jit
import sys
import os
#import sympy as sp
from scipy import linalg
import time
from numpy.polynomial.hermite import hermgauss
import argparse #I need to keep this 
import warnings
warnings.filterwarnings("error", category=RuntimeWarning, message="invalid value encountered in arctanh")

from numpy import cosh, tanh, arctanh, sin, cos, tan, arcsin, arccos, exp, array, sqrt, pi
#from sympy import *
from quadratures import gaussian_quadrature, trapezoidal_quadrature
from helper_functions import get_Guess_distribution,cosine4_mask
from mean_field_grid_rothe import *
np.set_printoptions(linewidth=300, precision=16, suppress=True, formatter={'float': '{:0.7e}'.format})
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DFT')))
from exchange_correlation_functionals import v_xc,epsilon_xc
import calculate_error_gaussian_removal as cegr
points=None
weights=None
# Function to save or append data
def v_ee_coulomb(grids):
    vp = 1 / np.sqrt(grids**2 + 1)
    return vp
def hartree_potential(grid,rho,weights,vee=v_ee_coulomb):
    v_h = np.zeros_like(rho)
    for i, xi in enumerate(grid):
        v_h[i] = np.sum(rho*vee(xi-grid)*weights)
    return v_h



@jit(nopython=True, fastmath=False,parallel=True)
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
    for i in range(num_gauss):
        indices_of_interest=np.where((np.abs(points-qvals[i])*avals[i])<6)
        funcvals, minus_half_laplacian_vals = gauss_and_minushalflaplacian(points[indices_of_interest], avals[i], bvals[i], pvals[i], qvals[i])

        functions[i][indices_of_interest] = funcvals
        minus_half_laplacians[i][indices_of_interest] = minus_half_laplacian_vals
    
    return functions, minus_half_laplacians
@jit(nopython=True, fastmath=False, parallel=True)
def setupfunctionsandDerivs(gaussian_nonlincoeffs, points):
    # Ensure parameters are in 2D form.
    if gaussian_nonlincoeffs.ndim == 1:
        avals = np.array([gaussian_nonlincoeffs[0]])
        bvals = np.array([gaussian_nonlincoeffs[1]])
        pvals = np.array([gaussian_nonlincoeffs[2]])
        qvals = np.array([gaussian_nonlincoeffs[3]])
    else:
        avals = gaussian_nonlincoeffs[:, 0]
        bvals = gaussian_nonlincoeffs[:, 1]
        pvals = gaussian_nonlincoeffs[:, 2]
        qvals = gaussian_nonlincoeffs[:, 3]
    num_gauss = avals.shape[0]
    nPoints = points.shape[0]
    
    # Allocate output arrays.
    zero_arrays = np.zeros((10, num_gauss, nPoints), dtype=np.complex128)
    fs, min_hal_lap, ader, bder, pder, qder, akin_der, bkin_der, pkin_der, qkin_der = zero_arrays    
    # Loop over Gaussians in parallel.
    for i in range(num_gauss):
        idx = np.nonzero((np.abs(points - qvals[i]) * avals[i]) < 6)[0]
        (fs[i, idx],min_hal_lap[i, idx],ader[i, idx],bder[i, idx],pder[i, idx],qder[i, idx],
         akin_der[i, idx],bkin_der[i, idx],pkin_der[i, idx],qkin_der[i, idx]) = gauss_and_minushalflaplacian_and_derivs(
        points[idx],avals[i], bvals[i], pvals[i], qvals[i])
    
    return (fs, min_hal_lap,ader, bder, pder, qder,akin_der, bkin_der, pkin_der, qkin_der)
def calculate_overlapmatrix(functions,wT):
    num_gauss=len(functions)
    overlap_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)
    for i in range(num_gauss):
        i_conj=np.conj(functions[i])
        for j in range(i,num_gauss):
            integrand_overlap=i_conj*functions[j]
            overlap_integraded=wT@integrand_overlap
            overlap_matrix[i,j]=overlap_integraded
            overlap_matrix[j,i]=np.conj(overlap_integraded)
    return overlap_matrix
#@jit(nopython=True,fastmath=False,cache=False)
def calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid,wT):
    num_gauss=len(functions)
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
    
    Returns
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
    functions,_=setupfunctions(gaussian_nonlincoeffs.reshape((C.shape[0],4)),points)
    return make_orbitals_numba(C,gaussian_nonlincoeffs,functions)

@jit(nopython=True,fastmath=False,cache=False)
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
def calculate_v_gauss(fockOrbitals,gaussian_nonlincoeffs,num_gauss,time_dependent_potential=None):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    return calculate_v_gauss_fast(np.array(fockOrbitals),num_gauss,time_dependent_potential,np.array(functions),np.array(minus_half_laplacians))

@jit(nopython=True,fastmath=False,cache=False)
def calculate_Fgauss_fast(fockOrbitals,num_gauss,time_dependent_potential,functions,minus_half_laplacians):
    nFock=len(fockOrbitals)
    Fgauss=minus_half_laplacians
    potential_term = potential_grid + time_dependent_potential
    electron_density=np.zeros(fockOrbitals.shape[1],dtype=np.complex128)
    for j in range(nFock):
        electron_density+=2*np.abs(fockOrbitals[j])**2
    coulomb_term=np.dot(electron_density,weighted_e_e_grid)
    Fgauss+=(potential_term+coulomb_term)*functions
    fock_orbitals_conj=np.conj(fockOrbitals)
    for i in range(num_gauss):
        for j in range(nFock):
            exchange_term =(fock_orbitals_conj[j] * functions[i]).T@weighted_e_e_grid
            Fgauss[i] += -exchange_term * fockOrbitals[j]
    return Fgauss
#@jit(nopython=True,fastmath=False,cache=False)
def calculate_v_gauss_fast(previous_time_orbitals,num_gauss,time_dependent_potential,functions,minus_half_laplacians):
    nOrbs=len(previous_time_orbitals)
    Agauss=minus_half_laplacians
    potential_term = potential_grid + time_dependent_potential
    electron_density=np.zeros(previous_time_orbitals.shape[1],dtype=np.complex128)
    for j in range(nOrbs):
        electron_density+=2*np.abs(previous_time_orbitals[j])**2
    coulomb_term=np.dot(electron_density,weighted_e_e_grid)
    potential_term+=coulomb_term
    potential_term+=v_xc(electron_density)
    Agauss+=potential_term*functions
    return Agauss
@jit(nopython=True,fastmath=False,cache=False)
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


class Rothe_evaluator:
    def __init__(self,old_params,old_lincoeff,time_dependent_potential,timestep,points,weights,number_of_frozen_orbitals=0,method="HF"):
        """
        old_params: The parameters for the Gaussians from the previous iteration
        old_lincoeff: The linear coefficients for the Gaussians in the basis of the old ones, from the previous iteration
        time_dependent_potential: The time-dependent potential evaluated at the relevant time
        timestep: The timestep used in the propagation
        """
        self.nbasis=old_lincoeff.shape[0]
        self.norbs=old_lincoeff.shape[1]
        self.method=method
        self.points=points
        self.weights=weights
        self.sqrt_weights=np.sqrt(weights)
        if method=="HF":
            self.orbital_operator=calculate_Fgauss_fast
            self.orbital_operator_slow=calculate_Fgauss
        elif method=="DFT":
            self.orbital_operator=calculate_v_gauss_fast
            self.orbital_operator_slow=calculate_v_gauss
        self.old_params=old_params
        self.old_lincoeff=old_lincoeff
        self.pot=time_dependent_potential
        self.dt=timestep
        self.nfrozen=number_of_frozen_orbitals
        self.params_frozen=old_params[:4*self.nfrozen]
        self.orbitals_that_represent_Fock=make_orbitals(self.old_lincoeff,self.old_params) #Orbitals that define the Fock operator; which are the old orbitals

        self.old_action=self.calculate_Adagger_oldOrbitals() #Essentially, the thing we want to approximate with the new orbitals
        self.f_frozen,self.fock_act_on_frozen_gauss=self.calculate_frozen_orbital_stuff()
        self.reshaped_sqrt_weights=self.sqrt_weights.reshape(-1, 1)
        self.new_action=None
    def calculate_Adagger_oldOrbitals(self):
        fock_act_on_old_gauss=self.orbital_operator_slow(self.orbitals_that_represent_Fock,self.old_params,num_gauss=self.nbasis,time_dependent_potential=self.pot) #Act with the OLD Fock operator on the OLD Gaussians
        Fock_times_Orbitals=calculate_Ftimesorbitals(self.old_lincoeff,fock_act_on_old_gauss)
        rhs=self.orbitals_that_represent_Fock-1j*self.dt/2*Fock_times_Orbitals
        return rhs
    def calculate_frozen_orbital_stuff(self):
        functions,minus_half_laplacians,_,_,_,_,_,_,_,_=setupfunctionsandDerivs(self.params_frozen.reshape((-1,4)),points)
        
        fock_act_on_frozen_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions),time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        return functions,fock_act_on_frozen_gauss
    def setupOneGaussianOptimization(self,nonlin_params_unfrozen):
        functions_u,minus_half_laplacians_u,_, _, _, _, _, _, _, _=setupfunctionsandDerivs(nonlin_params_unfrozen.reshape((-1,4)),self.points)
        fock_act_on_new_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions_u),time_dependent_potential=self.pot,
                                                    functions=np.array(functions_u),minus_half_laplacians=np.array(minus_half_laplacians_u))
        functions=np.concatenate((self.f_frozen,functions_u))
        fock_act_on_functions=np.concatenate((self.fock_act_on_frozen_gauss,fock_act_on_new_gauss))
        X=(functions+1j*self.dt/2*fock_act_on_functions).T
        X = X *self.reshaped_sqrt_weights
        self.new_action=X
    def optimize_oneGaussian(self,nonlin_params_unfrozen,index,return_grad=True,err_threshold=0.95):
        old_action=self.old_action *self.sqrt_weights
        gradient=np.zeros(4)
        functions_u,minus_half_laplacians_u,aderivs, bderivs, pderivs, qderivs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(nonlin_params_unfrozen,self.points)
        fock_act_on_new_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions_u),time_dependent_potential=self.pot,
                                                    functions=np.array(functions_u),minus_half_laplacians=np.array(minus_half_laplacians_u))
        X_i=functions_u+1j*self.dt/2*fock_act_on_new_gauss
        X_i = X_i *self.sqrt_weights
        if self.new_action is not None:
            X=self.new_action
        else:
            raise ValueError("new_action is None. Please call setupOneGaussianOptimization first.")
        X[:,index]=X_i[0,:]
        threshold=1e-8
        U,Sigma,Vdagger=np.linalg.svd(X,full_matrices=False)
        Sigma_invvals=np.where(np.abs(Sigma) > threshold, 1.0 / Sigma, 0.0)
        Sigma_inv=np.diag(Sigma_invvals)
        full_pseudoinverse=Vdagger.conj().T@Sigma_inv@U.conj().T
        
        new_lincoeff = full_pseudoinverse @ old_action.T  # shape (M, N)
        X_new = X @ new_lincoeff                           # shape (D, N)
        zs = old_action - X_new.T                          # shape (N, D)
        rothe_error = np.sum(np.linalg.norm(zs, axis=1)**2)

        overlapmatrix=X.T.conj()@X
        myslice=np.abs(overlapmatrix[index,:])
        for elem in myslice:
            if elem>err_threshold:
                rothe_error+=1e-3*(elem**2-err_threshold**2)/(1-err_threshold**2)
        if not return_grad:
            return rothe_error
        
        self.optimal_lincoeff=new_lincoeff
        function_derivs=np.array([aderivs[0],bderivs[0],pderivs[0],qderivs[0]])
        kin_derivs=np.array([aderiv_kin_funcs[0],bderiv_kin_funcs[0],pderiv_kin_funcs[0],qderiv_kin_funcs[0]])

        Fock_act_on_derivs=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(function_derivs),time_dependent_potential=self.pot,
                                                    functions=np.array(function_derivs),minus_half_laplacians=np.array(kin_derivs))
        Xders=(function_derivs+1j*self.dt/2*Fock_act_on_derivs).T
        
        Xders = Xders *self.reshaped_sqrt_weights
        tempmat=U@Sigma_inv@Vdagger
        zsT=zs.T
        for k in range(4):
            #See https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/
            Xder_k=Xders[:,k]
            col_index = self.nfrozen + k // 4
            c_k=new_lincoeff[col_index,:]
            Dkc_several = np.outer(Xder_k, c_k)
            ak_several=Dkc_several-U@(U.conj().T@Dkc_several)
            vector = Xder_k.conj() @ zsT  # or .T@ if you need exact shape details

            # Outer product -> shape (N, norbs)
            bk_several = np.outer(tempmat[:, col_index], vector)
            gradvecs = -ak_several - bk_several  # Shape: (N, num_orbitals)
            dot_products = np.sum(zs * gradvecs.conj().T, axis=1)  # Shape: (num_orbitals,)
            gradient[k] += 2 * np.real(np.sum(dot_products))  # Sum over all orbitals
        return rothe_error,gradient
    def rothe_plus_gradient(self,nonlin_params_unfrozen,printing=False,calculate_overlap=True,return_overlap=False):
        old_action=self.old_action *self.sqrt_weights
        gradient=np.zeros_like(nonlin_params_unfrozen)
        functions_u,minus_half_laplacians_u,aderivs, bderivs, pderivs, qderivs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(nonlin_params_unfrozen.reshape((-1,4)),self.points)
        fock_act_on_new_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions_u),time_dependent_potential=self.pot,
                                                    functions=np.array(functions_u),minus_half_laplacians=np.array(minus_half_laplacians_u))
        functions=np.concatenate((self.f_frozen,functions_u))
        fock_act_on_functions=np.concatenate((self.fock_act_on_frozen_gauss,fock_act_on_new_gauss))
        function_derivs=[]
        kin_derivs=[]
        for i in range(len(aderivs)):
            function_derivs+=[aderivs[i],bderivs[i],pderivs[i],qderivs[i]]
            kin_derivs+=[aderiv_kin_funcs[i],bderiv_kin_funcs[i],pderiv_kin_funcs[i],qderiv_kin_funcs[i]]
        function_derivs=np.array(function_derivs)
        kin_derivs=np.array(kin_derivs)
        X=(functions+1j*self.dt/2*fock_act_on_functions).T
        X = X *self.reshaped_sqrt_weights

        threshold=1e-7
        U,Sigma,Vdagger=np.linalg.svd(X,full_matrices=False)
        Sigma_invvals=np.where(np.abs(Sigma) > threshold, 1.0 / Sigma, 0.0)
        Sigma_inv=np.diag(Sigma_invvals)
        full_pseudoinverse=Vdagger.conj().T@Sigma_inv@U.conj().T
        
        new_lincoeff = full_pseudoinverse @ old_action.T  # shape (M, N)
        X_new = X @ new_lincoeff                           # shape (D, N)
        zs = old_action - X_new.T                          # shape (N, D)
        rothe_error = np.sum(np.linalg.norm(zs, axis=1)**2)
        
        self.optimal_lincoeff=new_lincoeff
        Fock_act_on_derivs=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(function_derivs),time_dependent_potential=self.pot,
                                                    functions=np.array(function_derivs),minus_half_laplacians=np.array(kin_derivs))
        Xders=(function_derivs+1j*self.dt/2*Fock_act_on_derivs).T
        
        Xders = Xders *self.reshaped_sqrt_weights
        M=len(nonlin_params_unfrozen)
        tempmat=U@Sigma_inv@Vdagger
        zsT=zs.T
        for k in range(M):
            #See https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/
            Xder_k=Xders[:,k]
            col_index = self.nfrozen + k // 4
            c_k=new_lincoeff[col_index,:]
            Dkc_several = np.outer(Xder_k, c_k)
            ak_several=Dkc_several-U@(U.conj().T@Dkc_several)
            vector = Xder_k.conj() @ zsT  # or .T@ if you need exact shape details

            # Outer product -> shape (N, norbs)
            bk_several = np.outer(tempmat[:, col_index], vector)
            gradvecs = -ak_several - bk_several  # Shape: (N, num_orbitals)
            dot_products = np.sum(zs * gradvecs.conj().T, axis=1)  # Shape: (num_orbitals,)
            gradient[k] += 2 * np.real(np.sum(dot_products))  # Sum over all orbitals        

        if calculate_overlap or return_overlap:
            overlapmatrix=calculate_overlapmatrix(functions,weights)
            
            overlapmatrix_eigvals=np.linalg.eigvalsh(overlapmatrix)
            ovlp_mindiag=overlapmatrix-np.eye(overlapmatrix.shape[0])
            ovlp_mindiag_unfrozen=ovlp_mindiag[self.nfrozen:,self.nfrozen:]
            max_indices=np.unravel_index(np.argmax(abs(ovlp_mindiag_unfrozen)),ovlp_mindiag_unfrozen.shape)
            max_indices=np.array(max_indices)
            max_indices[0]+=self.nfrozen
            max_indices[1]+=self.nfrozen
            max_indices=tuple(max_indices)
            smallest_eigvals=np.array2string(overlapmatrix_eigvals[:3], formatter={'float_kind': lambda x: "%.2e" % x})
            if printing:
                #print('\n'.join([' '.join([f'{elem:.1e}' for elem in row]) for row in np.abs(overlapmatrix)]))
                print("Smallest Overlap matrix eigenvalues",smallest_eigvals,
                        "Biggest Overlap matrix element", np.max(abs(ovlp_mindiag_unfrozen)), "at",max_indices)
        if return_overlap:
            return rothe_error,gradient, overlapmatrix,max_indices
        return rothe_error,gradient

    def rothe_error_oneremoved(self,nonlin_params_unfrozen):
        old_action=self.old_action *self.sqrt_weights
        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions,minus_half_laplacians,_,_,_,_,_,_,_,_=setupfunctionsandDerivs(nonlin_params.reshape((-1,4)),points)
        fock_act_on_new_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions),time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        X=functions+1j*self.dt/2*fock_act_on_new_gauss
        new_lincoeff=np.empty((self.nbasis,self.norbs),dtype=np.complex128)
        old_action=old_action
        X=X.T
        X = X *self.reshaped_sqrt_weights
        
        X_dag=X.conj().T
        XTX =X_dag @ X
        I=np.eye(XTX.shape[0])
        I_masked=np.eye(XTX.shape[0]-1)
        #I_masked[:self.nfrozen,:self.nfrozen]=0
        rothe_error=0
        zs=np.zeros_like(old_action)
        invmats=[]
        invmat=np.linalg.inv(XTX+ lambd * I)
        rothe_error_gaussian_removed=np.zeros(len(nonlin_params_unfrozen)//4)
        for orbital_index in range(old_action.shape[0]):
            
            Y=old_action[orbital_index]
            XTy = X_dag @ Y
            new_lincoeff[:,orbital_index]=invmat@XTy
            zs[orbital_index]=Y-X@new_lincoeff[:,orbital_index]
            rothe_error+=np.linalg.norm(zs[orbital_index])**2
            for i in range(len(self.params_frozen)//4,len(nonlin_params)//4):
                mask = np.arange(X.shape[1]) != i #Remove the i-th Gaussian
                X_masked=X[:,mask]
                X_dag_masked=X_masked.conj().T
                XTX_masked=X_dag_masked@X_masked
                invmat_masked=np.linalg.inv(XTX_masked+ lambd * I_masked)
                new_lincoeff_masked=invmat_masked@X_dag_masked@Y
                zs_masked=Y-X_masked@new_lincoeff_masked
                rothe_error_gaussian_removed[i-len(self.params_frozen)//4]+=np.linalg.norm(zs_masked)**2
        return rothe_error_gaussian_removed
    def rothe_error_oneOnly(self,other_nonlin_params):
        def opt_func(nonlin_params):
            all_nonlin_params=np.concatenate((other_nonlin_params,nonlin_params))
            error,gradient=self.rothe_plus_gradient(all_nonlin_params,False)
            return error,gradient[-4:]
        return opt_func
    
    def orthonormalize_orbitals(self,nonlin_params,old_lincoeff,orbital_norms=None):
        old_action=self.old_action *self.sqrt_weights
        functions,_=setupfunctions(nonlin_params.reshape((-1,4)),points)
        functions=functions.T
        functions2= functions*self.reshaped_sqrt_weights
        ovlp_matrix=np.conj(functions2.T)@functions2
        if orbital_norms is None:
            orbital_norms=np.ones(old_action.shape[0])
        ovlp_matrix_MO_basis = np.conj(old_lincoeff).T @ ovlp_matrix @ (old_lincoeff)
        # Diagonalize the overlap matrix to get eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(ovlp_matrix_MO_basis)
        S_pow12_inv = np.diag(eigvals**(-0.5))  # S^(-1/2)
        new_lincoeff =(old_lincoeff @ eigvecs @ S_pow12_inv @ eigvecs.T.conj())*np.sqrt(orbital_norms)
        self.optimal_lincoeff=new_lincoeff
        return new_lincoeff
    

def apply_mask(nonlin_params_old,lincoeff,nbasis,nfrozen,points,sqrt_weights):
    new_params=nonlin_params_old.copy()
    orbitals_before_mask=make_orbitals(lincoeff,nonlin_params_old)
    orbitals_masked=cosine_mask*orbitals_before_mask
    norbs=orbitals_before_mask.shape[0]
    nonlin_params_frozen=nonlin_params_old[:4*nfrozen]
    new_lincoeff_optimal=lincoeff.copy()
    def error_and_deriv(nonlin_params_new,return_best_lincoeff=False):
        nonlin_params=np.concatenate((nonlin_params_frozen,nonlin_params_new))
        functions,_,aderivs, bderivs, pderivs, qderivs, _, _, _, _=setupfunctionsandDerivs(nonlin_params.reshape((-1,4)),points)
        function_derivs=[]
        for i in range(nfrozen,len(aderivs)):
            function_derivs+=[aderivs[i],bderivs[i],pderivs[i],qderivs[i]]
        X=functions
        new_lincoeff=np.empty((nbasis,norbs),dtype=np.complex128)
        old_action=orbitals_masked*sqrt_weights
        X=X.T
        X = X *sqrt_weights.reshape(-1, 1)
        threshold=1e-7
        U,Sigma,Vdagger=np.linalg.svd(X,full_matrices=False)
        Sigma_invvals=np.where(np.abs(Sigma) > threshold, 1.0 / Sigma, 0.0)
        Sigma_inv=np.diag(Sigma_invvals)
        full_pseudoinverse=Vdagger.conj().T@Sigma_inv@U.conj().T
        tempmat=U@Sigma_inv@Vdagger
        new_lincoeff = full_pseudoinverse @ old_action.T  # shape (M, N)
        X_new = X @ new_lincoeff                           # shape (D, N)
        zs = old_action - X_new.T                          # shape (N, D)
        zsT=zs.T
        rothe_error = np.sum(np.linalg.norm(zs, axis=1)**2)

        Xders=np.array(function_derivs).T
        Xders = Xders *sqrt_weights.reshape(-1, 1)
        Xders=Xders
        gradient=np.zeros_like(nonlin_params_new)
        M=len(nonlin_params_new)
        for k in range(M):
            #See https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/
            Xder_k=Xders[:,k]
            col_index = nfrozen + k // 4
            c_k=new_lincoeff[col_index,:]
            Dkc_several = np.outer(Xder_k, c_k)
            ak_several=Dkc_several-U@(U.conj().T@Dkc_several)
            vector = Xder_k.conj() @ zsT  # or .T@ if you need exact shape details

            # Outer product -> shape (N, norbs)
            bk_several = np.outer(tempmat[:, col_index], vector)
            gradvecs = -ak_several - bk_several  # Shape: (N, num_orbitals)
            dot_products = np.sum(zs * gradvecs.conj().T, axis=1)  # Shape: (num_orbitals,)
            gradient[k] += 2 * np.real(np.sum(dot_products))  # Sum over all orbitals    
        if return_best_lincoeff:
            return rothe_error,gradient,new_lincoeff
        return rothe_error,gradient
    nit=0
    initial_mask_error,_,lincoeff_linear=error_and_deriv(nonlin_params_old[4*nfrozen:],True)
    mask_init_err=1e-10
    if initial_mask_error>mask_init_err and grid_b<grid_b_cancel:
        print("You should increase the grid size and rerun from a previous time step")
        sys.exit()
    error_due_to_mask=initial_mask_error
    
    if initial_mask_error>mask_init_err:
        solution,_,time,nit=minimize_transformed_bonds(error_and_deriv,
                                                        start_params=nonlin_params_old[4*nfrozen:],
                                                        gradient=True,
                                                        maxiter=50,
                                                        gtol=1e-14,
                                                        both=True,
                                                        lambda_grad0=1e-14)
        new_params=np.concatenate((nonlin_params_frozen,solution))
        error_due_to_mask,grad,new_lincoeff_optimal=error_and_deriv(solution,True)
        if error_due_to_mask>initial_mask_error:
            new_params=nonlin_params_old
            new_lincoeff_optimal=lincoeff_linear
            error_due_to_mask=initial_mask_error
        print("Niter: %d, Error due to mask: %.2e/%.2e, time: %.1f"%(nit,error_due_to_mask,initial_mask_error,time))
        functions,_=setupfunctions(new_params.reshape((-1,4)),points)
        functions=functions.T
        functions2= functions*sqrt_weights.reshape(-1, 1)
        ovlp_matrix=np.conj(functions2.T)@functions2

        ovlp_matrix_MO_basis=np.conj(new_lincoeff_optimal).T@ovlp_matrix@new_lincoeff_optimal
        eigvals,eigvecs=np.linalg.eigh(ovlp_matrix_MO_basis)
        new_lincoeff_optimal=new_lincoeff_optimal@eigvecs #Orthogonalize

        return new_params,new_lincoeff_optimal,eigvals
    else: 
        return nonlin_params_old,lincoeff,None
class Rothe_propagation:
    def __init__(self,params_initial,lincoeffs_initial,pulse,timestep,points,weights,nfrozen=0,t=0,norms=None,params_previous=None,method="HF"):
        self.nbasis=lincoeffs_initial.shape[0]
        self.norbs=lincoeffs_initial.shape[1]
        self.method=method
        self.points=points
        self.weights=weights
        self.sqrt_weights=np.sqrt(weights)
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
        self.last_added_t=0
    def propagate(self,t,maxiter,last_added=False):
        added_recently=abs(self.t-self.last_added_t)<=0.3
        allow_adding=True
        initial_lincoeffs=self.lincoeffs
        initial_params=self.params
        self.time_dependent_potential=self.pulse(t+self.dt/2)*points
        initial_full_new_params=initial_params[4*self.nfrozen:]
        rothe_evaluator=Rothe_evaluator(initial_params,initial_lincoeffs,self.time_dependent_potential,
                                        dt,number_of_frozen_orbitals=self.nfrozen,method=self.method,points=self.points,weights=self.weights)
        initial_rothe_error,_=rothe_evaluator.rothe_plus_gradient(initial_full_new_params)
        print("Initial Rothe error: %e"%sqrt(initial_rothe_error))
        start_params=initial_full_new_params
        ls=np.linspace(0,1,6)
        threshold=0.97
        best=0
        if self.adjustment is not None and self.adjustment.shape == initial_full_new_params.shape:
            updated_res = [initial_rothe_error]
            dx = self.adjustment
            bounds = [(avals_min, avals_max),(bvals_min, bvals_max),(pvals_min, pvals_max),(muvals_min, muvals_max)]
                    
            for ls_val in ls[1:]:
                changed = initial_full_new_params + ls_val * dx
                # Process every block of 4 parameters
                for j in range(len(changed) // 4):
                    for k, (lower, upper) in enumerate(bounds):
                        idx = j * 4 + k
                        changed[idx] = min(max(changed[idx], lower), upper) 
                updated_re, _, overlap_matrix, _= rothe_evaluator.rothe_plus_gradient(changed,return_overlap=True)
                biggest_overlap_element=np.max(np.abs(overlap_matrix-np.eye(overlap_matrix.shape[0])))
                updated_re = 1e100 if biggest_overlap_element > threshold else updated_re
                updated_res.append(updated_re)
            
            best = np.argmin(updated_res)
            start_params = initial_full_new_params + ls[best] * dx
            initial_rothe_error = updated_res[best]
            print("Old Rothe error, using change of %.1f: %e" % (ls[best], sqrt(initial_rothe_error)))
        else:
            print("Old Rothe error: %e"%sqrt(initial_rothe_error))
        gtol=1e-14
        initial_rothe_error,grad0,overlap_matrix, max_indices=rothe_evaluator.rothe_plus_gradient(start_params,return_overlap=True)
        inold=initial_rothe_error
        biggest_val=np.abs(overlap_matrix[max_indices])
        
        """
        if biggest_val>threshold:
            #max_indices=np.arange(self.nfrozen,len(overlap_matrix),1)
            print("We first optimize the two Gaussians that are closest to each other, independently of the rest:",max_indices,biggest_val)

            for mxidx in [max_indices[0]]:
               
                rothe_evaluator.setupOneGaussianOptimization(start_params)
                parameters_to_optimize=start_params[4*(mxidx-self.nfrozen):4*((mxidx-self.nfrozen+1))]
                err,grad=rothe_evaluator.optimize_oneGaussian(parameters_to_optimize,mxidx,return_grad=True)
                #hess_inv0=np.eye(4)
                sol=minimize(rothe_evaluator.optimize_oneGaussian,
                            parameters_to_optimize,
                            method='L-BFGS-B',
                            args=(mxidx,False,threshold),
                            options={'maxiter':200,'gtol':1e-16},
                            bounds=[(avals_min, avals_max),(bvals_min, bvals_max),(pvals_min, pvals_max),(muvals_min, muvals_max)],
                    )
                solution=sol.x
                err=sol.fun
                #print("New parameters: ",solution)
                start_params[4*(mxidx-self.nfrozen):4*((mxidx-self.nfrozen+1))]=solution
            initial_rothe_error,grad0,overlap_matrix, _=rothe_evaluator.rothe_plus_gradient(start_params,return_overlap=True)
            print("New: %.3e, Old: %.3e"%(sqrt(initial_rothe_error),sqrt(inold)))
        """
        if optimize_untransformed:
            hess_inv0=np.diag(1/(abs(grad0)+self.lambda_grad0))
            sol=minimize(rothe_evaluator.rothe_plus_gradient,
                         start_params,jac=True,
                         method='BFGS',
                         options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':gtol})
            solution=sol.x
            niter=sol.nit
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            print("Number of iterations: ",sol.nit)
            new_rothe_error,_=rothe_evaluator.rothe_plus_gradient(solution)
            print("Rothe Error after optimization: %e using lambd=%.1e"%(sqrt(new_rothe_error),self.lambda_grad0))
            print(list(sol.x))
        else:
            intervene=True
            multi_bonds=0.1
            maxiter_to_use=maxiter
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(rothe_evaluator.rothe_plus_gradient,
                                                        start_params=start_params,
                                                        gradient=True,
                                                        maxiter=maxiter_to_use,
                                                        gtol=gtol,
                                                        both=True,
                                                        multi_bonds=multi_bonds,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=intervene,
                                                        write_file=True,
                                                        timepoint=self.t)
            
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            new_rothe_error,_=rothe_evaluator.rothe_plus_gradient(solution,calculate_overlap=True,printing=True)
            backup_re=sqrt(new_rothe_error)
            backup_solution=solution.copy()
            backup_lincoeff=new_lincoeff.copy()
            print("RE after opt: %.3e/%.3e, Ngauss=%d, time=%.1f, niter=%d/%d"%(sqrt(new_rothe_error),rothe_epsilon_per_timestep,len(solution)//4,time,niter,maxiter))
        sqrt_RE=sqrt(new_rothe_error)
        if  (initial_rothe_error<0.99*new_rothe_error and niter>0 and self.t>20) or (sqrt_RE>1.05*rothe_epsilon_per_timestep) or (niter==0 and self.t>20):
            print("Rothe error increased; OR bigger than acceptable, something needs to be done")
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(rothe_evaluator.rothe_plus_gradient,
                                                        start_params=initial_full_new_params,
                                                        gradient=True,
                                                        maxiter=maxiter,
                                                        gtol=gtol*1e-1,
                                                        both=True,
                                                        multi_bonds=multi_bonds,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=False)
       
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            sqrt_RE=sqrt(new_rothe_error)
            if sqrt_RE>backup_re:
                print("Rothe error increased after second optimization. We will try to optimize from the first solution, we will keep the first solution")
                solution=backup_solution
                new_lincoeff=backup_lincoeff
                sqrt_RE=backup_re
            else:
                print("RE after opt: %.2e/%.2e, Ngauss=%d, time=%.1f, niter=%d/%d"%(sqrt(new_rothe_error),rothe_epsilon_per_timestep,len(solution)//4,time,niter,maxiter))
        smallest_re_onerem=1e10
        errors_oneremoved=rothe_evaluator.rothe_error_oneremoved(solution)
        small_i=np.argmin(errors_oneremoved)
        smallest_re_onerem=np.min(errors_oneremoved)
        #small_i=28-20
        #smallest_re_onerem=1e-8
        initial_rothe_error,grad0,overlap_matrix, max_indices=rothe_evaluator.rothe_plus_gradient(solution,return_overlap=True)
        smallest_eigval=np.linalg.eigvalsh(overlap_matrix)[0]
        print("Errors upon removal: ", np.array2string(np.sort(np.sqrt(errors_oneremoved)), formatter={'float_kind': lambda x: "%.2e" % x}))
        print("Smallest eigval: ",smallest_eigval)
        removal =False
        unnecessary_gaussian=(smallest_eigval<1e-8 and smallest_re_onerem<rothe_epsilon_per_timestep)
        if smallest_re_onerem<new_rothe_error*1.1 or unnecessary_gaussian:
            removal =True
        toobig=sqrt_RE>rothe_epsilon_per_timestep
        num_gauss_total_extra=len(solution)//4
        run_refitting=True
        #removal=True if smallest_eigval<1e-8 else removal
        if self.t>10 and not added_recently and maxiter>0 and  (toobig or removal) and num_gauss_total_extra<100:
            print("Rothe error is too big, or a Gaussian can be reoptimized, or eigval_problem")
            self.last_added_t=self.t
            
            if removal:
                print("Removing Gaussian %d with error %.3e"%(small_i,sqrt(errors_oneremoved[small_i])))
                solution_removed=np.delete(solution,[small_i*4,small_i*4+1,small_i*4+2,small_i*4+3])
                self.nbasis-=1
                rothe_evaluator.nbasis=self.nbasis
                solution,new_rothe_error,time,niter=minimize_transformed_bonds(rothe_evaluator.rothe_plus_gradient,
                                                        start_params=solution_removed,
                                                        gradient=True,
                                                        maxiter=maxiter,
                                                        gtol=gtol,
                                                        both=True,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=False)
                new_lincoeff=rothe_evaluator.optimal_lincoeff
                print("Rothe error after basis reoptimization: %e"%sqrt(new_rothe_error))
            removal=False
            self.nbasis+=1
            rothe_evaluator.nbasis=self.nbasis
            new_params_list=[]
            rothe_errors=[]
            avals=solution[0::4];bvals=solution[1::4];pvals=solution[2::4];muvals=solution[3::4]
            minvals=np.array([avals_min,bvals_min,pvals_min,muvals_min])
            maxvals=np.array([avals_max,bvals_max,pvals_max,muvals_max])
            multipliers_min=np.array([1.05,0.95,0.95,0.95])
            number_of_randoms=500
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
            print("Rothe error before optimization: %e"%sqrt(rothe_errors[best]))
            opt_func_temp=rothe_evaluator.rothe_error_oneOnly(new_params_list[best][:-4])
            _,grad=opt_func_temp(new_params_list[best][-4:])
            hess_inv0=np.diag(1/abs(grad+self.lambda_grad0*np.array(len(grad))))
            sol=minimize(opt_func_temp,new_params_list[best][-4:],jac=True,method='BFGS',options={'maxiter':50,'gtol':gtol,"hess_inv0":hess_inv0})
            very_best=sol.x
            very_best[0]=abs(very_best[0])
            new_rothe_error,_=opt_func_temp(very_best)
            after_first_lin=rothe_evaluator.optimal_lincoeff.copy()
            after_first_RE=sqrt(new_rothe_error)
            maxiterino=500 if not unnecessary_gaussian else 0
            print("Rothe error after first optimization: %e"%sqrt(new_rothe_error))
            best_new_params=np.concatenate((new_params_list[best][:-4],very_best))
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(rothe_evaluator.rothe_plus_gradient,
                                                        start_params=best_new_params,
                                                        gradient=True,
                                                        maxiter=maxiterino,
                                                        gtol=gtol,
                                                        both=True,
                                                        multi_bonds=1,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=False)
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            print("Rothe error after optimization: %e"%sqrt(new_rothe_error))
            sqrt_RE=sqrt(new_rothe_error)
            if sqrt_RE>after_first_RE:
                print("Rothe error increased after second optimization. We will try to optimize from the first solution")
                solution,new_rothe_error,time,niter=minimize_transformed_bonds(rothe_evaluator.rothe_plus_gradient,
                                                        start_params=new_params_list[best],
                                                        gradient=True,
                                                        maxiter=maxiterino,
                                                        gtol=gtol,
                                                        both=True,
                                                        multi_bonds=1,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=False)
                new_lincoeff=rothe_evaluator.optimal_lincoeff
                print("Rothe error after yet another optimization: %e"%sqrt(new_rothe_error))
                sqrt_RE=sqrt(new_rothe_error)
            if sqrt_RE>after_first_RE:
                print("Rothe error increased after third optimization. We will use the single-optimized Gaussian only")
                sqrt_RE=after_first_RE
                solution=best_new_params
                new_lincoeff=after_first_lin
        self.last_rothe_error=sqrt_RE
        new_params=np.concatenate((initial_params[:4*self.nfrozen],solution))
        #At this point, we are done with the optimization, and we have the new parameters. Now we have to "fix" the overlap matrix
        final_rothe_error,_,overlap_matrix, max_indices=rothe_evaluator.rothe_plus_gradient(solution,return_overlap=True)
        ovlpmatrix_eigvals,_=np.linalg.eigh(overlap_matrix)
        smallest=ovlpmatrix_eigvals[0]
        overlap_matrix_non=overlap_matrix[self.nfrozen:,self.nfrozen:]
        biggest_overlap_element=np.max(np.abs(overlap_matrix_non-np.eye(overlap_matrix_non.shape[0])))
        index=None
        eigval_issue=False
        if smallest<1e-8:
            print("An eigenvalue of the overlap matrix is too small: %e"%smallest)
            eigval_issue=True
            removal_eigvals,largest_elements=cegr.full_diagonalization_smallest_eigenvalues(overlap_matrix,self.nfrozen)
            largest_elements_sorted=np.argsort(largest_elements)
            if removal_eigvals[largest_elements_sorted[0]]>5e-8 and largest_elements[largest_elements_sorted[0]]>0.98:
                maxindex=largest_elements_sorted[0]
            elif removal_eigvals[largest_elements_sorted[1]]>5e-8 and largest_elements[largest_elements_sorted[1]]>0.98:
                maxindex=largest_elements_sorted[1]
            else:
                maxindex=np.argmax(removal_eigvals)
            print("Removing Gaussian with index %d leads to a smallest overlap of %e"%(maxindex,removal_eigvals[maxindex]))
            index=maxindex
        changed_basis=False
        final_cost=1e100
        if (eigval_issue or biggest_overlap_element>0.99) and run_refitting:
            print("Biggest overlap element: %e, eigvalue issue: %s"%(biggest_overlap_element,eigval_issue))
            num_new=0
            optimized_params=[]
            costs=[]
            while sqrt(final_cost)>2*rothe_epsilon_per_timestep and num_new<3:
                num_new+=1
                gaussianremover=cegr.GaussianRemovalCalculator(new_lincoeff,new_params.reshape((-1,4)),molecule=molecule,penalty_constant=1e-2)
                optimized_nonlincoeff, updated_lincoeffs, largest_overlap, final_cost ,num_extra= gaussianremover.run(
                    5,500,index=index,num_new=num_new,best_threshold=rothe_epsilon_per_timestep**2,max_ovlp_first=0.7, max_ovlp_second=0.98)
                optimized_params.append([optimized_nonlincoeff,updated_lincoeffs,num_extra])
                costs.append(final_cost)
                print("Largest overlap after removal", largest_overlap)
                print("Final cost: %e/%e"%(sqrt(final_cost),rothe_epsilon_per_timestep))
                
            if sqrt(final_cost)>rothe_epsilon_per_timestep:
                best_index=np.argmin(costs)
                optimized_nonlincoeff=optimized_params[best_index][0]
                updated_lincoeffs=optimized_params[best_index][1]
                final_cost=costs[best_index]
                num_extra=optimized_params[best_index][2]
            self.nbasis+=num_extra-1
            rothe_evaluator.nbasis=self.nbasis

            new_lincoeff=updated_lincoeffs
            new_params=optimized_nonlincoeff.flatten()
        """
        elif biggest_overlap_element>0.99:
            print("Biggest overlap element: %e, eigvalue issue: %s"%(biggest_overlap_element,eigval_issue))
            num_new=-1
            optimized_params=[]
            costs=[]
            while sqrt(final_cost)>rothe_epsilon_per_timestep and num_new<2:
                num_new+=1
                gaussianremover=cegr.GaussianRemovalCalculator(new_lincoeff,new_params.reshape((-1,4)),molecule=molecule)
                optimized_nonlincoeff, updated_lincoeffs, largest_overlap, final_cost ,num_extra= gaussianremover.run(20,500,eigval_issue=eigval_issue,index=index,num_new=num_new)
                optimized_params.append([optimized_nonlincoeff,updated_lincoeffs,num_new])
                costs.append(final_cost)
                print("Largest overlap after removal", largest_overlap)
                print("Final cost: %e/%e"%(sqrt(final_cost),rothe_epsilon_per_timestep))
            if sqrt(final_cost)>rothe_epsilon_per_timestep:
                best_index=np.argmin(costs)
                optimized_nonlincoeff=optimized_params[best_index][0]
                updated_lincoeffs=optimized_params[best_index][1]
                final_cost=costs[best_index]
                num_new=optimized_params[best_index][2]
            self.nbasis+=num_new
            rothe_evaluator.nbasis=self.nbasis

            new_lincoeff=updated_lincoeffs
            new_params=optimized_nonlincoeff.flatten()
        """
        #After opatimization: Make sure orbitals are orthonormal, and apply mask
        new_lincoeff=rothe_evaluator.orthonormalize_orbitals(new_params,new_lincoeff,self.norms)
        if optimize_untransformed:
            print("New Lincoeff:")
            print(list(new_lincoeff))
        new_params,new_lincoeff,new_norms=apply_mask(new_params,new_lincoeff,self.nbasis,self.nfrozen,self.points,self.sqrt_weights)
        if new_norms is not None:
            self.norms=new_norms
        solution=new_params[4*self.nfrozen:]

        #Reorthogonalize the orbitals, but nor reorthonormalize
        new_lincoeff=rothe_evaluator.orthonormalize_orbitals(new_params,new_lincoeff,self.norms)

        C_flat=new_lincoeff.flatten()
        linparams_new=np.concatenate((C_flat.real,C_flat.imag))
        self.full_params=np.concatenate((linparams_new,new_params))
        try:
            self.adjustment=solution-initial_full_new_params
        except ValueError:
            len_init=len(initial_full_new_params)
            self.adjustment=solution[:len_init]-initial_full_new_params
            self.adjustment=np.concatenate((self.adjustment,np.zeros(len(solution)-len_init)))
        self.params=new_params
        self.lincoeffs=new_lincoeff
        avals=self.params[4*self.nfrozen::4]
        bvals=self.params[4*self.nfrozen+1::4]
        pvals=self.params[4*self.nfrozen+2::4]
        muvals=self.params[4*self.nfrozen+3::4]
        print("Avals: [%.3e, %.3e]; Bvals: [%.3e, %.3e]; Pvals: [%.3e, %.3e]; Muvals: [%.3e, %.3e]"%(np.min(avals),np.max(avals),np.min(bvals),np.max(bvals),np.min(pvals),np.max(pvals),np.min(muvals),np.max(muvals)))
        print("Norms: ",self.norms)
        print("All positions:", np.sort(muvals))
        #print("All parameters")
        #print(list(self.params))
        #print("All linear coefficients")
        #print(list(self.lincoeffs))
    def propagate_nsteps(self,Tmax,maxiter):
        filename="WF_%s_%s_%.4f_%d_%d_%d_%.3e.npz"%(self.method,molecule,E0,initlen,num_gauss,maxiter,rothe_epsilon)
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
        while abs(self.t)<Tmax:
            if abs(self.t-self.last_added_t)<=0.3 and self.t>1:
                self.propagate(np.real(self.t),maxiter,last_added=True) #If a gaussian was recently added, we need to optimize more
            else:
                self.propagate(np.real(self.t),maxiter)
            self.t+=abs(self.dt)
            #F,S=calculate_Fock_and_overlap(self.lincoeffs,self.params,time_dependent_potential=self.time_dependent_potential)
            x_expectation=calculate_x_expectation(self.lincoeffs,self.params)
            save_wave_function(filename, self.full_params, self.dt,x_expectation,self.t,self.last_rothe_error,self.norms,self.nbasis,self.norbs)
            
    def plot_orbitals(self,t):
        plt.figure()
        orbitals=make_orbitals(self.lincoeffs,self.params)
        for i in range(self.norbs):
            plt.plot(points,np.abs(orbitals[i]),label="|Orbital %d|"%i)
        plt.legend()
        plt.yscale("log")
        plt.close()
if __name__=="__main__":
    n_extra=4
    F_input = float(sys.argv[1])# Maximum field strength in 10^14 W/cm^2
    maxiter=int(sys.argv[2])
    start_time=float(sys.argv[3])
    molecule=sys.argv[4]
    if molecule=="LiH":
        initlen=20
    elif molecule=="LiH2":
        initlen=34
    freeze_start=sys.argv[5]
    rothe_epsilon=float(sys.argv[6])
    optimize_untransformed=sys.argv[7] #Should be False unless we optimize initial state with imaginary time propagation
    if optimize_untransformed != "True":
        optimize_untransformed=False
    method=sys.argv[8]
    if freeze_start=="freeze":
        nfrozen=initlen
    else:
        nfrozen=0

    inner_grid=10
    if F_input==1:
        grid_b_cancel=200
        grid_b=200
    elif F_input==4:
        grid_b=150
        grid_b_cancel=400
    elif F_input==0:
        grid_b=30
        grid_b_cancel=30
    grid_a=-grid_b
    muvals_max=grid_b-10
    muvals_min=grid_a+10
    avals_min=0.1
    avals_max=2
    bvals_min=-5
    bvals_max=5
    pvals_min=-20
    pvals_max=20
    minimize_transformed_bonds=make_minimizer_function(avals_min,avals_max,bvals_min,bvals_max,pvals_min,pvals_max,muvals_min,muvals_max)
    points_inner,weights_inner=gaussian_quadrature(-inner_grid,inner_grid,24*inner_grid+1)
    grid_spacing=0.4
    num_points=int((grid_b-inner_grid)/grid_spacing)
    points_outer1,weights_outer1=trapezoidal_quadrature(grid_a, -inner_grid, num_points)
    points_outer2,weights_outer2=trapezoidal_quadrature(inner_grid, grid_b, num_points)
    points=np.concatenate((points_outer1,points_inner,points_outer2))
    weights=np.array(np.concatenate((weights_outer1,weights_inner,weights_outer2)),dtype=np.float64)
    n=len(points)
    lambd=1e-16
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
    lincoeff_initial=None

    if initlen==20 and molecule=="LiH" and method=="HF":
        gaussian_nonlincoeffs=[np.float64(0.7369315282821801), np.float64(0.060943240217486296), np.float64(-0.9696177241214851), np.float64(-1.2592582961667025), np.float64(0.35718667190182646), np.float64(0.020428064054160706), np.float64(-0.3223321138648322), np.float64(-1.1109308993493665), np.float64(0.957067723034581), np.float64(-0.29109004131968275), np.float64(-0.32034755142081883), np.float64(-1.0161554287230574), np.float64(1.6658844123687666), np.float64(0.9579330308999002), np.float64(-1.5852278448016168), np.float64(-0.4040913874456688), np.float64(0.2582845741578324), np.float64(-0.0013916488910019644), np.float64(-0.22762168761456897), np.float64(-0.2693610710551233), np.float64(0.45961763303381015), np.float64(-0.03826469557754684), np.float64(-0.34479893034801795), np.float64(-0.06366634727433937), np.float64(0.43931932017351266), np.float64(0.004639436092001097), np.float64(0.024695748708846922), np.float64(-0.343650879973623), np.float64(2.397242859434365), np.float64(2.51955263308398), np.float64(-0.11269439938309442), np.float64(-0.5985713818066646), np.float64(0.5770347002721897), np.float64(-0.10130384087379886), np.float64(-0.42701417232737454), np.float64(-0.2342685961771616), np.float64(0.21114423399064677), np.float64(-0.0015183248872974399), np.float64(-0.11626645097557121), np.float64(-0.18072463952271675), np.float64(2.2353325132044173), np.float64(-1.4734164606056073), np.float64(-1.3858661708444189), np.float64(-0.355406555825135), np.float64(0.6582388715838225), np.float64(0.7502227763343051), np.float64(-4.290449685716489), np.float64(-2.7367000835607347), np.float64(1.6575554383752933), np.float64(-1.4259747642022074), np.float64(-2.229493777016462), np.float64(0.15023678612223557), np.float64(0.3228303052011197), np.float64(0.005953941136364994), np.float64(-0.04895292729637613), np.float64(-0.04759368440746367), np.float64(1.8487317163757329), np.float64(1.1479421162756858), np.float64(-0.5905772149140904), np.float64(0.16325182157880322), np.float64(1.0450749487863762), np.float64(-0.450370664241363), np.float64(-1.042002307412309), np.float64(-0.21090785116347083), np.float64(0.9822154044046787), np.float64(-0.05349456053767097), np.float64(0.21300667978739266), np.float64(1.2426561726622698), np.float64(1.9355165285505425), np.float64(-2.5469573673305366), np.float64(-1.6028810976777825), np.float64(0.147243460098322), np.float64(1.4593966657542938), np.float64(1.135216509366876), np.float64(-2.623995054262496), np.float64(-0.6171851480938769), np.float64(1.1919146218937111), np.float64(-0.2664092461249298), np.float64(-0.5812711512455832), np.float64(0.1459248802583415)]
        lincoeff_initial=[array([-0.2950925320020051+0.0557030544004624j,  0.5133852880982396+0.5785671591378152j]), array([0.1136685699418282-0.0043298319047353j, 0.3158756650380747+0.1983317502914557j]), array([ 0.1366706589126782-0.5951790809566253j, -0.7950371531332456+0.5244005816238181j]), array([0.2114095964415541+0.3104241124591035j, 0.0044021187387148-0.3792393012849827j]), array([0.0265218678931399-0.0212166636724208j, 0.1000223367910638-0.0200990169607749j]), array([-0.0923065813241979-0.1723274810381845j,  0.4000403032219623-0.5938019710421287j]), array([-0.07720301026171  -0.0851200705944271j, -0.5362313335434507+0.3431157376294222j]), array([-0.0025898963562976+0.0001930419381076j,  0.0020690892156754+0.0015767598049143j]), array([-0.0298737220474603-0.2172383671346852j,  0.0014147585045106-0.7114876501727779j]), array([0.0041843090765277-0.00682771677575j  , 0.0198962245898795-0.0108794398970367j]), array([-0.0010881487606998+0.0085168099706184j,  0.0158261343487511-0.0015426346577732j]), array([ 0.0012944717549586-0.0010016757681633j, -0.0023158140879679-0.0003783267658676j]), array([0.0096712511754571+0.0386278850970437j, 0.0277424516347078+0.0087865422453223j]), array([-0.0690668310330724-0.1116791705030292j, -0.0419059365840657-0.353953262423501j ]), array([0.0210854821824963+0.0246215116412038j, 0.0012933700973773-0.0618240104114665j]), array([-0.4261478837422948-0.1757540974584176j,  0.2093144644061359+0.5865313954374654j]), array([-0.0032407522542861-0.005125577737997j , -0.0063675042692439-0.0089102818170481j]), array([0.0000282703568964-0.00234280406727j  , 0.0010480067545816-0.0100551739227366j]), array([ 0.1357692452255108-0.0993282638586122j, -0.1874245259268906+0.0394036044555611j]), array([ 0.181632403054413 +0.0509449097776507j, -0.3919187235555229+0.1685375662243869j])]

    elif initlen==34 and molecule=="LiH2" and method=="HF":
        gaussian_nonlincoeffs=[np.float64(2.3006401335443254), np.float64(2.419627487988478), np.float64(-0.6892709833789855), np.float64(-3.4755861943182906), np.float64(2.4442867728535096), np.float64(0.35526945711430064), np.float64(-0.1732694073157898), np.float64(1.8453811746303619), np.float64(0.7387517320397643), np.float64(-0.42577350479421905), np.float64(-0.14513683038551822), np.float64(-1.7857493780979286), np.float64(0.9864022191223563), np.float64(-0.30384756265807183), np.float64(-0.5236406518985024), np.float64(2.7146619280976454), np.float64(0.685273735781145), np.float64(0.13861952672822744), np.float64(3.785683224106183), np.float64(2.1727557559570903), np.float64(1.7832286953931105), np.float64(-1.4434199625018724), np.float64(-2.5624805587740997), np.float64(-2.390224797367677), np.float64(2.320720372421454), np.float64(0.011947289940559167), np.float64(-1.9955825972131922), np.float64(2.34612580020303), np.float64(2.333653471547741), np.float64(-0.9706430408400013), np.float64(-1.7162287131486162), np.float64(-3.4266197688145823), np.float64(0.600689456585156), np.float64(0.04688444142364901), np.float64(-0.7964724320271611), np.float64(1.8945898631476232), np.float64(1.5887191110409158), np.float64(1.2717289536934018), np.float64(-2.459046702917216), np.float64(-3.467432372674582), np.float64(2.1276760629823004), np.float64(-1.6144602156624897), np.float64(-0.8566997242342776), np.float64(-2.325034698570429), np.float64(0.990050385609722), np.float64(-0.8229049988971069), np.float64(-0.7998682739182552), np.float64(-2.0100209754541796), np.float64(1.8844601293021304), np.float64(1.9384280240122116), np.float64(-2.1085786822883383), np.float64(1.789109341578251), np.float64(0.8239177474888838), np.float64(0.778767153500414), np.float64(-1.2466348555468727), np.float64(1.1124709223302403), np.float64(1.5631154977785808), np.float64(-1.9705948360627696), np.float64(-0.7268638888740965), np.float64(3.0013047352431794), np.float64(1.1051037390169545), np.float64(-0.16675095498015027), np.float64(0.04314914514638123), np.float64(-3.1556351879363578), np.float64(1.0183100397142006), np.float64(-0.5000110292578452), np.float64(1.3148216147922585), np.float64(1.0413792675666647), np.float64(1.5567806148971295), np.float64(0.9786386956959157), np.float64(-0.6642873939655879), np.float64(-3.1435314856369265), np.float64(0.7646607589450125), np.float64(-0.21994203363391876), np.float64(1.05226957831923), np.float64(-2.949387484585731), np.float64(0.657943602900683), np.float64(-0.8921051348345146), np.float64(1.1212738766721098), np.float64(0.6176273879096035), np.float64(1.3566409799835777), np.float64(1.2054568804715204), np.float64(-0.5060499023271738), np.float64(2.680986656543605), np.float64(0.6660164276425797), np.float64(0.14027150944864014), np.float64(0.15573777333094638), np.float64(1.9902307924423663), np.float64(0.32472467801464816), np.float64(-0.025520092463880022), np.float64(0.5725675310331696), np.float64(-2.260009553659389), np.float64(0.8615991469522906), np.float64(0.21738867015992186), np.float64(1.352315996089856), np.float64(-1.8865622471856913), np.float64(0.34275055346633176), np.float64(-0.022604340038030774), np.float64(-0.36678309709733864), np.float64(1.8600337175398218), np.float64(0.4671874045107948), np.float64(-0.029494875291995806), np.float64(0.3569208883408571), np.float64(-3.0757189932360087), np.float64(0.6305413808052774), np.float64(-0.2124221066068054), np.float64(-0.8107370118747668), np.float64(3.763204523147577), np.float64(0.19656542795643275), np.float64(-0.0008776046594668322), np.float64(-0.09481967588369856), np.float64(0.46674606968676957), np.float64(0.22849978703338125), np.float64(-0.009977237949789785), np.float64(0.24202624913520224), np.float64(-2.3796903825457623), np.float64(0.6105642920762232), np.float64(0.001251047992436387), np.float64(0.009080847607482), np.float64(-3.092712790653501), np.float64(0.25325542117549277), np.float64(-0.08912634037401566), np.float64(1.5577453925230162), np.float64(-4.361940584951661), np.float64(0.22087892381383042), np.float64(0.009366529964754922), np.float64(-0.1578669166180696), np.float64(-1.7917159118051849), np.float64(0.3636232723107457), np.float64(0.05470495914459701), np.float64(2.428530774845887), np.float64(1.9848090819228676), np.float64(0.2568666152670833), np.float64(0.02298533755998491), np.float64(0.2580752277578288), np.float64(0.6839726221711965)]
        lincoeff_initial=[array([-0.0031944857206451+0.0048257854195362j, -0.0028173123318757+0.002524539588385j , -0.0026919869310758+0.0001513290424519j,  0.0032166795416315+0.0023731841479517j]), array([-0.0034022517394194-0.0009310767114645j,  0.0021901943239479-0.0004083288282796j,  0.0046309639927858+0.0009838477261683j,  0.0037729390560648+0.0003334982426869j]), array([-0.083972932096842 +0.08084810505947j  , -0.2185696704944604+0.1491100526305942j, -0.1315309431330566+0.2924979078858719j,  0.0824891907115638-0.300916492232738j ]), array([-0.189511929922747 +0.2445497754927522j,  0.2733376273019569-0.2721441659690797j, -0.1436432357372895+0.0788090870776391j, -0.1747065403723622-0.0621020673379842j]), array([-0.0009139762696156+0.0032845908838371j, -0.0000055862618857-0.0037765683527373j,  0.0037731467315694-0.0020147183050092j,  0.0011273568033686-0.0001410562890213j]), array([-0.0236042975856844+0.0184229218534744j, -0.0179859089175481+0.0121945019653672j, -0.0090463965812976+0.002926099348657j ,  0.0052705238106056+0.0094788805833627j]), array([ 0.002835296772724 +0.0000712104219641j,  0.0002938593887149+0.001684495590985j , -0.0087925251977403-0.0002234917422107j, -0.0056460105503177-0.0000850634617556j]), array([-0.0036483986282267-0.0010267213346841j, -0.0033287824432174-0.001303068044069j , -0.0021972306308619-0.0005675973792102j,  0.0008857972892413+0.000859792663138j ]), array([ 0.5357835932050034+0.2187037475044637j, -0.6975617452284087-0.1652633234696158j,  0.2221064005486789-0.2851826147680241j, -0.0176648718162568+0.0896578381170334j]), array([0.0494906915910231+0.0026971132535383j, 0.0356628853054433-0.0000087158909911j, 0.0107659719022989+0.0015471837983466j, 0.0091016493123717-0.0126090610014049j]), array([-0.0006055245613144+0.000835565410369j ,  0.0006576956209712-0.0009656745168139j, -0.0011586220768396-0.0026452968341111j,  0.0020024119073843+0.003631182018111j ]), array([-0.0016394780309984+0.0762959936316667j, -0.0532863746142034+0.0660010467625858j, -0.0686720356022378+0.0702832339362491j,  0.0955485917234724-0.0603168367413446j]), array([ 0.0026232028507343+0.002297314704643j ,  0.0002074922425284+0.0033548749338207j, -0.0123756143100963-0.0091924612982202j, -0.0076613284492145-0.0061509038716244j]), array([-0.1189023055813204+0.004631045084266j ,  0.1381520274779189-0.0332169039168178j,  0.0011490341932564+0.0718587281069998j, -0.0005222973817727-0.0124170121206005j]), array([ 0.0132132306656305-0.0043440616324248j, -0.0158675384048175+0.0053545640272732j,  0.0044962558391728-0.0023390514992338j,  0.002438049670597 +0.0050832492813328j]), array([-0.195851823983265 +0.4972132899006508j, -0.3161235676959051+0.6991451524916946j,  0.0020112840818442+0.7007341033283231j, -0.0668576670732504-0.5581492679992262j]), array([ 0.0819365012663447+0.015167941189952j , -0.0671757587462011+0.0079630802484926j, -0.0629163186473779-0.0298583300037112j, -0.0559190464947816-0.0012528308173834j]), array([-0.0275769470343716+0.1646211924412198j, -0.03464433645262  +0.0894946318900504j, -0.0672910289695846+0.0150863022680338j,  0.1142201697193262+0.0506705926452924j]), array([ 0.7770097835612594-0.3160020107832421j,  0.4681997088783619-0.3649462888823372j, -0.0243676336006154-0.1106495892068824j,  0.2975249165607736-0.2397009216515817j]), array([-0.0119350203263495-0.0012268764706568j,  0.0144477718652268+0.0009790834846165j, -0.0057309130834527+0.000078728841603j ,  0.0005899521906984-0.0048902415459964j]), array([ 0.0315966458887013-0.0272931271433275j, -0.0352550806673119+0.0382733044830636j, -0.005312052640588 -0.0202584637262866j,  0.0090108327519815+0.0032709783395067j]), array([-0.8082182406184488+0.2386037467035516j,  0.6397640833401641-0.3444323452747445j,  0.5826701832932554+0.0330806083417124j,  0.3881042708102095-0.3470374869379947j]), array([ 0.0663842263785725-0.2099883354248412j, -0.033541787838914 +0.5429843461394023j,  0.4645485038809394+0.2363209825566481j, -0.6181764750248554-0.3237649977882119j]), array([ 0.0556390576551484-0.3218022809606765j, -0.134865415636287 -0.2034575644717815j, -0.1616424000278519+0.1156678031081108j,  0.0988784282686989-0.3233760912527431j]), array([ 0.4380210037929789-0.2704805853580797j, -0.5485056808638215+0.3429659272758762j,  0.103713377567065 -0.1682413829633961j,  0.1530725980888409+0.1359631049147658j]), array([0.177193971017891 -0.3116310650317843j, 0.1631599440987232-0.4199270956741304j, 0.0290929102646065-0.4134105351359151j, 0.0387657671778858+0.339537576228966j ]), array([ 0.0765247375066828-0.1282779201835943j, -0.0987882051725469+0.1562455540937803j,  0.0179229632171724-0.0572859941333008j,  0.0705728000156372+0.0250487015618437j]), array([ 0.1074647870303741-0.0485982136298657j, -0.1313872658903988+0.0599475589258747j, -0.0000866155539637-0.029622533309382j ,  0.0456522992645394+0.0139941390239962j]), array([ 0.0282068129379003-0.0159314192533427j, -0.0526425166105729+0.0236083280978691j,  0.0866385569894758-0.0706788739330372j, -0.0698023952111204+0.0717727313732032j]), array([-0.4854745851708375-0.6202351131886755j, -0.4943725420472897-0.6108414840037755j, -0.3340740899415017-0.2922846511866505j,  0.0662096879119304+0.1864565836684536j]), array([ 0.0052433316579481+0.0039020180742947j, -0.0044171295268181-0.0076948822696239j, -0.0027433600924527-0.0024241883446555j,  0.0064522623825169+0.0047337109541716j]), array([-0.0052553420517669+0.0688516293667958j,  0.0180407708452083-0.1007586189049702j,  0.1135393396357256-0.0150505604424428j, -0.1005476571960437+0.0639637001630009j]), array([ 0.0057435970107704-0.0123705639718795j, -0.0077782772834101+0.0153509032473325j,  0.0013070137860105-0.0056892110816057j,  0.0065441150995673+0.0012975785177348j]), array([-0.5181176052500773-0.4127563733890762j,  0.6473965095635413+0.4930770589967639j, -0.1216250451849095-0.0125643071415425j, -0.0221569968671651-0.1757119500134398j])]

    elif initlen==20 and molecule=="LiH" and method=="DFT":
        gaussian_nonlincoeffs=[np.float64(0.8770667448629443), np.float64(0.39825176300883014), np.float64(-1.8555175709478406), np.float64(-0.9367215963420926), np.float64(0.3905462179744208), np.float64(0.026585394672052336), np.float64(-0.1878759730642028), np.float64(-1.3129339414389976), np.float64(1.242003550943037), np.float64(-0.7979239927032846), np.float64(-0.12399764663023753), np.float64(-1.2545153449385666), np.float64(2.157844102196907), np.float64(-0.6192279829195073), np.float64(0.5748147850524854), np.float64(-0.9849036339726925), np.float64(0.26342678008178627), np.float64(-0.0013743299104635618), np.float64(-0.2406328406221324), np.float64(-0.15891550363179402), np.float64(0.49685352838620167), np.float64(0.09427465912847734), np.float64(-0.4543748747020159), np.float64(-0.687171816180345), np.float64(0.5185622571628867), np.float64(-0.03233303308575061), np.float64(-0.327129771717043), np.float64(-0.18935395674444785), np.float64(1.9952067700681029), np.float64(1.6395333724831669), np.float64(1.2536806457325993), np.float64(-0.3251381239072825), np.float64(0.5500154614609288), np.float64(0.15702998179259867), np.float64(0.03674719779949898), np.float64(-0.1485924934607752), np.float64(0.20165947355531072), np.float64(0.0014099279502278763), np.float64(-0.12727273567961958), np.float64(-0.34227583242809084), np.float64(2.1646972200911976), np.float64(0.16173008237746547), np.float64(0.7714133956323795), np.float64(-0.7770995056830476), np.float64(0.6725679392074336), np.float64(0.5558865603746446), np.float64(-2.6094399762075846), np.float64(-1.761766036131598), np.float64(1.7025143906989832), np.float64(-1.3711986771455347), np.float64(-0.6284267261438575), np.float64(0.6968435677927118), np.float64(0.34664971163094527), np.float64(0.01518621787725737), np.float64(0.0247581172776112), np.float64(-0.5085399795109118), np.float64(1.5565153098833173), np.float64(1.3502743227256782), np.float64(0.875116840818846), np.float64(-0.17024814158479493), np.float64(0.9705566602000737), np.float64(-0.3893552178989144), np.float64(-2.3458527322588503), np.float64(-0.665074866259308), np.float64(0.9665870051573374), np.float64(-0.7068465095734476), np.float64(-2.3330274928559245), np.float64(0.20910362462253626), np.float64(1.2672173006008802), np.float64(-1.436696898995686), np.float64(-2.186841305556244), np.float64(0.31487028317702587), np.float64(1.4998292414479786), np.float64(1.9783528521148797), np.float64(-0.9655653902492066), np.float64(-1.1246212448762756), np.float64(0.8599068768200502), np.float64(-0.1164232196730801), np.float64(-3.9440504234023024), np.float64(-0.4114264656011302)]
        lincoeff_initial=[array([-0.2968214123952604+0.441749221148022j, -1.1915129754172644+0.904470657682361j]), array([ 0.1741663071727461-0.324330814809734j , -0.505580600315648 -0.4059243397633829j]), array([-0.0138333374411233-0.3467707968416037j,  0.5738426789699321-0.4854632270801221j]), array([0.0631487894610813-0.0256651262183117j, 0.0190890411110766-0.2420524223804169j]), array([-0.0503567325594961-0.0724931982197052j, -0.1401712616759223+0.0756151441779205j]), array([0.1825622461001588-0.0641121487411909j, 0.244908102872328 -1.0812385729822633j]), array([-0.5744151130285768+0.1341133335746678j,  0.2427278461818254+1.0493379710010307j]), array([ 0.006068858374343 -0.0872448326665859j, -0.0334978587249085-0.0086854847659208j]), array([0.093087615474584 +0.0836519135069623j, 0.7159297012510142+0.1479861797604927j]), array([-0.0034857363848977-0.0154286288155572j, -0.0279640267360905+0.0030698839647894j]), array([-0.0427039076182657-0.2013896025950431j, -0.2687054926345595-0.0047386324471578j]), array([-0.0165795492665882+0.0100898503356429j,  0.0225364115909476+0.0736298497023236j]), array([-0.0041573341788653+0.0116928310548266j,  0.0239691264488773+0.0068564502348153j]), array([-0.3464318419543209+0.2353631481871738j,  0.3775811654929917+0.6540283483976452j]), array([-0.1715123057298013-0.0980949833134842j, -0.0606258229053992-0.1098486928486695j]), array([-0.2020975139468071-0.2514457839659731j, -0.0534281732850369-1.0258976505652018j]), array([ 0.0746036566020434-0.0907669378727558j, -0.1591812844182409-0.1082149576774316j]), array([0.0573972097490129+0.0645674527215206j, 0.1284874801010172-0.1099636263063288j]), array([-0.0041714211487198+0.0032691460836437j, -0.0131438332867257+0.007861901636017j ]), array([0.052180652253699 -0.0943348390689933j, 0.1293857651042383-0.3279377238415726j])]

    elif initlen==34 and molecule=="LiH2" and method=="DFT":
        gaussian_nonlincoeffs=[np.float64(2.2828611170922177), np.float64(1.506255366300714), np.float64(-0.7809612896074882), np.float64(-3.822767198083051), np.float64(2.420829271079871), np.float64(-0.0472727278178807), np.float64(1.3173593880445817), np.float64(1.722771914343222), np.float64(1.4691322071598305), np.float64(-2.270317117563174), np.float64(-0.11229245747788673), np.float64(-4.132418172548109), np.float64(1.048536159438172), np.float64(-0.5647465896166154), np.float64(-0.4826836518001277), np.float64(2.6917777754215533), np.float64(0.6469217112209793), np.float64(0.3171451130753678), np.float64(3.7924054354635364), np.float64(2.7416135245452415), np.float64(1.1949373059164015), np.float64(-1.8399816574681775), np.float64(-1.3655775864405648), np.float64(-1.3758778886839105), np.float64(2.071113494988917), np.float64(2.324993132902627), np.float64(1.961666604919854), np.float64(2.250082588223965), np.float64(1.965574267683791), np.float64(-0.510538880765004), np.float64(-1.5551527737504216), np.float64(-3.4770997247843938), np.float64(0.847469107212752), np.float64(0.09251404388984734), np.float64(0.16239468950995903), np.float64(1.849518246481416), np.float64(1.8590079519875853), np.float64(1.9787082654647101), np.float64(-1.6347550804558422), np.float64(-4.085506806710172), np.float64(1.8713317226573858), np.float64(-1.8898471857981225), np.float64(-1.0237412558219592), np.float64(-2.35743008999497), np.float64(1.034113719769682), np.float64(-0.732843336831841), np.float64(-0.18635280930741782), np.float64(-2.0720069270862846), np.float64(2.2696127337696885), np.float64(-1.463546047408439), np.float64(1.8557895029993852), np.float64(1.4503753471550074), np.float64(0.8053538046608519), np.float64(0.6432673247854955), np.float64(-1.2149730835117736), np.float64(1.4548339844150031), np.float64(1.6038363886186988), np.float64(-0.29236767953856796), np.float64(0.08548203080152561), np.float64(3.1608019011335653), np.float64(0.9394237577612452), np.float64(-0.40583273611223253), np.float64(0.37719076015745157), np.float64(-2.8650119133744787), np.float64(1.387476246746981), np.float64(0.6019978313079889), np.float64(0.059944737600477076), np.float64(0.8267396154704296), np.float64(1.3727921210804246), np.float64(1.2009260838974554), np.float64(-0.6902832085137005), np.float64(-3.363579982830093), np.float64(0.813014196869663), np.float64(-0.14574983671308897), np.float64(1.2199250558606662), np.float64(-2.289480319956812), np.float64(0.7127674644630012), np.float64(-0.7171478104021142), np.float64(1.5681080373582645), np.float64(0.7097572197741638), np.float64(1.3301004141649586), np.float64(0.8716251480607429), np.float64(-0.23771114855193426), np.float64(2.869748497030881), np.float64(0.7145074627353148), np.float64(-0.10999017701330673), np.float64(-0.7176141198750853), np.float64(2.6574907382970188), np.float64(0.44203694247507963), np.float64(-0.0022124292424398687), np.float64(0.7651100068884105), np.float64(-3.282863881956371), np.float64(0.6799408563230877), np.float64(0.20889438901550025), np.float64(2.044748343794848), np.float64(-1.9286305880821906), np.float64(0.3589140242152011), np.float64(0.038580447025534474), np.float64(0.5259852145382388), np.float64(2.0771575173934145), np.float64(0.5102726983703989), np.float64(0.0037304795881782215), np.float64(0.4825058943230659), np.float64(-3.3416113208068947), np.float64(0.5438299769215936), np.float64(-0.16783988922774615), np.float64(-1.0665873122376879), np.float64(3.0294895517698595), np.float64(0.17176710729682942), np.float64(-0.001952212142941003), np.float64(-0.10846163062067211), np.float64(0.5186258806803985), np.float64(0.25291142849684495), np.float64(-0.04742161356971026), np.float64(0.826178178262354), np.float64(-4.297027704035078), np.float64(0.6829858903437335), np.float64(-0.019682547549399), np.float64(0.38649121964213096), np.float64(-3.198449583540473), np.float64(0.3537256440279481), np.float64(-0.1411818674489324), np.float64(1.889849266338731), np.float64(-4.109856558857817), np.float64(0.22811518280611043), np.float64(-0.005640655343800769), np.float64(0.19236679304775667), np.float64(-2.4560693154715394), np.float64(0.6270991062888217), np.float64(-0.27879831993084503), np.float64(1.1046017227570277), np.float64(0.786904494435626), np.float64(0.2582740166449906), np.float64(0.029644348865087922), np.float64(0.43619554490059026), np.float64(2.1247091279237327)]
        lincoeff_initial=[array([-0.0038764075540985+0.0093432023019409j,  0.0083329482244411-0.0027405750337994j, -0.0015104527470092-0.0011995431806777j,  0.0144678895017119-0.003495108352012j ]), array([-0.0026322482239468+0.0056576134107227j,  0.0024520093225537-0.0026023774840312j,  0.0128982955960628+0.0242662778554794j, -0.0042323763448557+0.00149029426191j  ]), array([-0.0003431127717697+0.0015061269095501j,  0.0007523613620904-0.0008227927042121j, -0.000516353630925 -0.0001488504125093j, -0.0000164297995745+0.0002110083675463j]), array([ 0.6453371263211243+0.4883026829625685j, -0.024444417663373 +0.8247561446307282j,  0.0311838963243282+0.4791108598729203j, -0.1861925508358502+0.2009059110294165j]), array([ 0.0331055980135061+0.0033435558584938j,  0.0172621100524298+0.0296282239749325j,  0.0151431015841686+0.0152858485433386j, -0.0016537455002969+0.0112267576746845j]), array([ 0.0036463396944433+0.007094760113954j , -0.0039577479413331-0.0055887190934109j, -0.003970954087445 +0.0021278267651197j, -0.0052366773140003+0.0020585407105317j]), array([-0.0008545204124931+0.0005212925159824j,  0.0015596700900214+0.0001293497929605j, -0.0021523687380216+0.0070349379357825j, -0.001178893517771 -0.0003503858612967j]), array([ 0.0281087543990376-0.0023924431848461j, -0.0282907099154291-0.0105905275078964j, -0.0043504379671162+0.0111050187599424j, -0.0233583420585322+0.0317292946441649j]), array([-0.6535270797636727-0.1295627081822199j, -0.2083248624560101-0.8693371943014616j,  0.6619453906287296+0.1650759187228935j,  0.0411130131782568-0.0638454222651012j]), array([ 0.0231375293336701+0.0031613067422963j, -0.016007384373602 -0.0149622105240852j, -0.0035009121818217+0.0062975335687716j, -0.0187216951122039-0.0055918018712933j]), array([ 0.0022632469674992+0.0058118034007634j,  0.0024380874075233-0.0080396204616317j, -0.0022989520068248-0.0011060730955433j, -0.0055360165701359-0.0013005808737881j]), array([ 0.2888419261869878-0.1918727830480951j, -0.3526487152096682+0.0272152036233035j,  0.0286473585130697+0.1320836099234181j,  0.1214863702311924+0.1879184072107888j]), array([-0.0030545581987544+0.000345667558604j ,  0.0011448475257143+0.0011640732245448j, -0.0075418726370851+0.0100689390551944j, -0.0013972835890457-0.0015642230893348j]), array([ 0.6241354028789667-0.0001259532312468j,  0.4080970335358027+0.4493774868213278j,  0.4083704673323325+0.3400814324154527j, -0.0292909804620791+0.2317618505845055j]), array([-0.1139410470758764+0.0308468103963978j, -0.0992668370315667-0.0677566882546202j, -0.075724554114967 -0.0500311439419632j, -0.0047001311048735-0.0419205051177117j]), array([-0.6343765650965548-1.031682261871676j ,  0.0034088263792074+0.887668553899397j ,  0.3601403099004423-0.2062387138641823j, -0.3493586171556327-0.4264773032681541j]), array([-0.0562000752371601-0.1209098198485244j,  0.0738796375280502-0.1140856461891954j,  0.0209836387668671-0.0807321894845705j,  0.0425980717180499-0.0189033449105815j]), array([ 0.0775582745192913+0.0564207972878269j, -0.0487517386629665-0.0724555933225713j, -0.0309823108667714+0.0264364192962799j, -0.0553939410988045+0.0502222183677496j]), array([ 0.3083820140384793+0.2216167186412929j, -0.1895095524053875-0.3104719943060023j, -0.1465339991342088+0.0862370935655846j, -0.8014373951970944+0.4785932274067824j]), array([ 0.0332220409011112+0.1923074243658604j, -0.1279512171809237+0.1536704219715658j, -0.0877623348536632+0.1131178453699481j, -0.0683221290589316+0.0019283768250172j]), array([-0.1731772261513903-0.2358296918869194j,  0.0541021230523885-0.2766767327690405j,  0.0119721098373444-0.2504141044540902j,  0.0962952535341979-0.0533432703488797j]), array([0.7762273698554822-0.5414153448301146j, 0.9424813907458902+0.096071548553089j , 0.7905999546235064+0.238152389338993j , 0.1150348687550764+0.3314154206000076j]), array([-0.1349719481749082+1.2560587924251558j,  0.7067991303958563-0.7225057587430009j, -0.3714487975520324-0.1354714342996177j, -1.0953889732174715-0.4030167263085351j]), array([-0.3121770767991886+0.0339900318303194j,  0.2422872728988299+0.1376123604719976j,  0.0180244615182694-0.0925781812616428j,  0.2091797760270214+0.0445879072811774j]), array([-0.8930692445262738+0.0641116463044391j, -0.6266979324796362-0.6346701761293475j, -0.62049513579544  -0.4369832995564989j,  0.0105233202298429-0.335943115829907j ]), array([ 0.2854600172552072+0.777415921072846j ,  0.1229270267906429-0.4478948403254753j, -0.2263423268632934+0.1577547294462438j,  0.5425822227843163+0.2685690746383253j]), array([ 0.0628379723035061+0.3911349842236411j, -0.1800037162965247+0.303530113345327j , -0.19094570609831  +0.2999206183423412j, -0.1562066984397818-0.0105915767204075j]), array([0.0329360045449693-0.0247470181008024j, 0.0274030230475655+0.0182746114290204j, 0.0318723238403936+0.0061427362159844j, 0.0103793165888159+0.0154694345443026j]), array([-0.0989357047349382+0.0735396324206782j,  0.0725178659645491-0.008485294267049j , -0.0282376946879604-0.0324959973484686j, -0.0222649570794259-0.0710482231477039j]), array([ 0.2147274364119787-0.8163055264508459j, -0.4142019488046657+0.4479392935650412j,  0.3238861828367228+0.107821968033725j ,  1.6594832884379334-0.6405342073124998j]), array([ 0.0753557842992135-0.2403730770857549j, -0.1540925191573732+0.1011335570899299j,  0.0687680825709992+0.0306801340350249j,  0.1354939121573607+0.0977520508989263j]), array([-0.4090685274729827+0.0335646973347747j,  0.2174516702251606+0.0522127780925799j, -0.0345758685397843-0.1609417518082383j,  0.0477892297379245-0.2473562405651686j]), array([0.1667458895762516-0.4813245912459077j, 0.5589201876985224-0.0931613506108223j, 0.3690408458245873-0.0037474815674972j, 0.1619569498580145+0.0741918274388819j]), array([-0.1996842590967241+0.1106577217748874j, -0.1742996800105324-0.1152910483508679j, -0.1815890645583128-0.0483242223933542j, -0.0417419316639579-0.0842094930725038j])]

    else:
        raise ValueError("initlen not supported")

    gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs)
    gaussian_nonlincoeffs[::4]=abs(gaussian_nonlincoeffs[::4]) # a-values can be positive or negative, but we want them to be positive

    gaussian_nonlincoeffs=list(np.array(gaussian_nonlincoeffs).reshape(-1,4))
    if molecule=="LiH":
        a=5
    elif molecule=="LiH2" and method=="HF":
        a=13
    elif molecule=="LiH2" and method=="DFT":
        a=13
    nextra_half=n_extra//2
    pos_list=np.linspace(a,a+2*(nextra_half-1),nextra_half)
    pos_list=np.concatenate((-pos_list,pos_list))
    for k in range(len(pos_list)):
        params=[1/sqrt(2),0,0,pos_list[k]]
        gaussian_nonlincoeffs.append(params)
        lincoeff_initial.append(np.array([0]*norbs))

    lincoeff_initial=np.array(lincoeff_initial)
    gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs)
    num_gauss=gaussian_nonlincoeffs.shape[0]
    potential_grid=calculate_potential(Z_list,R_list,alpha,points)


    onebody_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)
    overlap_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)

    #time_dependent_potential=0.1*points #I. e. 0.1*x - very strong field
    run_HF=True if (start_time<0.05) else False #If we are starting from the beginning, we need to run HF to get the initial state
    if run_HF and method=="HF":
        E,crap,morecrap=calculate_energy(gaussian_nonlincoeffs,return_all=True,maxiter=1,C_init=lincoeff_initial)
        x_expectation_t0=calculate_x_expectation(lincoeff_initial,gaussian_nonlincoeffs)
        print("Initial energy of %s:%.10f"%(molecule,E))
    E0=F0 =np.sqrt(F_input/(3.50944758*1e2))  # Maximum field strength

    omega = 0.06075  # Laser frequency
    t_c = 2 * np.pi / omega  # Optical cycle
    n_cycles = 3
    dt=0.05#*(-1j)
    td = n_cycles * t_c  # Duration of the laser pulse
    tfinal = td  # Total time of the simulation
    try:
        nsteps=int(tfinal/dt)
    except TypeError:
        nsteps=1000
    rothe_epsilon_per_timestep=rothe_epsilon/nsteps
    t=np.linspace(0,tfinal,1000)
    fieldfunc=laserfield(E0, omega, td)

    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs.reshape((num_gauss,4)),points)

    Tmax=tfinal
    filename="WF_%s_%s_%.4f_%d_%d_%d_%.3e.npz"%(method,molecule,E0,initlen,num_gauss,maxiter,rothe_epsilon)
    print("Filename:",filename)
    gaussian_nonlincoeffs_prev=None
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
    #gaussian_nonlincoeffs_prev=None

    if tmax==0 and method=="DFT":

        orbitals=make_orbitals_numba(lincoeff_initial,gaussian_nonlincoeffs,functions)
        print(calculate_overlapmatrix(orbitals,weights))

        kin_orbitals=make_orbitals_numba(lincoeff_initial,gaussian_nonlincoeffs,minus_half_laplacians)
        
        kinetic_energy=2*sum([np.dot(np.conj(weights*orbitals[i]),kin_orbitals[i]) for i in range(norbs)]) #Factor 2 because each orbital is counted twice
        density=2*sum([orbitals[i]*np.conj(orbitals[i]) for i in range(norbs)]) #Factor 2 because each orbital is counted twice
        potential_energy=np.sum(density*weights*potential_grid)  
        exchange_correlation_energy=sum(density*weights*epsilon_xc(density))
        hartree_pot=hartree_potential(points,density,weights)
        hartree_energy=0.5*sum(density*hartree_pot*weights)
        repulsion_energy=0
        for i in range(len(Z_list)):
            for j in range(i+1,len(Z_list)):
                repulsion_energy+=Z_list[i]*Z_list[j]/np.abs(R_list[i]-R_list[j])
        energy=kinetic_energy+potential_energy+exchange_correlation_energy+hartree_energy+repulsion_energy
        print("Kinetic energy:",kinetic_energy)
        print("Potential energy:",potential_energy)
        print("Exchange-correlation energy:",exchange_correlation_energy)
        print("Hartree energy:",hartree_energy)
        print("Repulsion energy:",repulsion_energy)

        print("Initial DFT energy: %f"%energy)
    rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,timestep=dt,points=points,weights=weights,
                                      nfrozen=nfrozen,t=tmax,norms=norms_initial,params_previous=gaussian_nonlincoeffs_prev,method=method)
    rothepropagator.propagate_nsteps(Tmax,maxiter)
