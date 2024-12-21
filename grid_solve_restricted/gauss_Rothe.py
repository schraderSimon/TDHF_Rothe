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

from numpy import cosh, tanh, arctanh, sin, cos, tan, arcsin, arccos, exp, array, sqrt, pi
#from sympy import *
from quadratures import gaussian_quadrature, trapezoidal_quadrature
from helper_functions import get_Guess_distribution,cosine4_mask
from mean_field_grid_rothe import *
np.set_printoptions(linewidth=300, precision=6, suppress=True, formatter={'float': '{:0.4e}'.format})
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DFT')))
from exchange_correlation_functionals import v_xc,epsilon_xc
# Function to save or append data



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
        avals,bvals,pvals,qvals=[[i] for i in gaussian_nonlincoeffs[:4]]
    else:
        avals,bvals,pvals,qvals=[gaussian_nonlincoeffs[:,i] for i in range(4)]
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
#@jit(nopython=True,fastmath=True,cache=False)
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
"""
class Hartree_Fock_grid():
    def __init__(self,functions,minus_half_laplacians,potential_grid,weights,e_e_grid):
        self.functions=functions
        self.minus_half_laplacians=minus_half_laplacians
        self.potential_grid=potential_grid
        self.weights=weights
        self.num_gauss=len(functions)
        self.e_e_grid=e_e_grid
    @jit(nopython=True,fastmath=True,cache=False)
    def calculate_onebody_and_overlap(self):
        num_gauss=self.num_gauss
        onebody_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)
        overlap_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)
        for i in range(num_gauss):
            i_conj=np.conj(self.functions[i])
            for j in range(i,num_gauss):
                integrand_minus_half_laplace=i_conj*self.minus_half_laplacians[j]
                integrand_overlap=i_conj*self.functions[j]
                integrand_potential=integrand_overlap*self.potential_grid
                overlap_integraded=self.weights@integrand_overlap
                ij_integrated=self.weights@integrand_minus_half_laplace+self.weights@integrand_potential
                onebody_matrix[i,j]=ij_integrated
                onebody_matrix[j,i]=np.conj(ij_integrated)
                overlap_matrix[i,j]=overlap_integraded
                overlap_matrix[j,i]=np.conj(overlap_integraded)
        return onebody_matrix,overlap_matrix
    @jit(nopython=True,fastmath=True,cache=False)
    def calculate_twobody_integrals(self):
        twobody_integrals = np.zeros((self.num_gauss,self.num_gauss,self.num_gauss,self.num_gauss), dtype=np.complex128)
        cross_functions = np.zeros((self.num_gauss, self.num_gauss, len(self.functions[0])), dtype=np.complex128)
        weighted_e_e_grid = self.e_e_grid * self.weights[:, np.newaxis]
        conj_functions=np.conj(self.functions)
        for i in range(self.num_gauss):
            for j in range(self.num_gauss):
                cross_functions[i, j] = conj_functions[i] * self.functions[j]
        for i in range(self.num_gauss):
            for k in range(i+1):
                index_ik=i*self.num_gauss+k
                ik_e_contr = np.sum(cross_functions[i, k][:, np.newaxis] * weighted_e_e_grid, axis=0)
                for j in range(self.num_gauss):
                    for l in range(self.num_gauss):
                        index_jl=j*self.num_gauss+l
                        if index_ik<index_jl:
                            continue
                        jl_grid = cross_functions[j, l]
                        val=np.sum(jl_grid * ik_e_contr * self.weights)
                        cval=np.conj(val)
                        twobody_integrals[i, k, j, l] = val
                        twobody_integrals[j, l, i, k] = val
                        twobody_integrals[k, i, l, j] = cval
                        twobody_integrals[l, j, k, i] = cval
        return twobody_integrals
"""
#@jit(nopython=True,fastmath=True,cache=False)
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
def calculate_v_gauss(fockOrbitals,gaussian_nonlincoeffs,num_gauss,time_dependent_potential=None):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    return calculate_v_gauss_fast(np.array(fockOrbitals),num_gauss,time_dependent_potential,np.array(functions),np.array(minus_half_laplacians))

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
    for i in range(num_gauss):
        for j in range(nFock):
            exchange_term =(fock_orbitals_conj[j] * functions[i]).T@weighted_e_e_grid
            Fgauss[i] += 2 * coulomb_terms[j] *functions[i]-exchange_term * fockOrbitals[j]
    return Fgauss
#@jit(nopython=True,fastmath=True,cache=False)
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


class Rothe_evaluator:
    def __init__(self,old_params,old_lincoeff,time_dependent_potential,timestep,number_of_frozen_orbitals=0,method="HF"):
        """
        old_params: The parameters for the Gaussians from the previous iteration
        old_lincoeff: The linear coefficients for the Gaussians in the basis of the old ones, from the previous iteration
        time_dependent_potential: The time-dependent potential evaluated at the relevant time
        timestep: The timestep used in the propagation
        """
        self.nbasis=old_lincoeff.shape[0]
        self.norbs=old_lincoeff.shape[1]
        self.method=method
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
        
    def calculate_Adagger_oldOrbitals(self):
        fock_act_on_old_gauss=self.orbital_operator_slow(self.orbitals_that_represent_Fock,self.old_params,num_gauss=self.nbasis,time_dependent_potential=self.pot) #Act with the OLD Fock operator on the OLD Gaussians
        Fock_times_Orbitals=calculate_Ftimesorbitals(self.old_lincoeff,fock_act_on_old_gauss)
        rhs=self.orbitals_that_represent_Fock-1j*self.dt/2*Fock_times_Orbitals
        return rhs
    def calculate_frozen_orbital_stuff(self):
        functions,minus_half_laplacians,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(self.params_frozen.reshape((-1,4)),points)
        
        fock_act_on_frozen_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions),time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        return functions,fock_act_on_frozen_gauss
    
    def rothe_plus_gradient(self,nonlin_params_unfrozen,hessian=False,printing=False,calculate_overlap=False):
        old_action=self.old_action *sqrt_weights
        gradient=np.zeros_like(nonlin_params_unfrozen)
        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions_u,minus_half_laplacians_u,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(nonlin_params_unfrozen.reshape((-1,4)),points)
        fock_act_on_new_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
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
        Fock_act_on_derivs=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
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
        if calculate_overlap:
            overlapmatrix=calculate_overlapmatrix(functions,weights)
            overlapmatrix_eigvals=np.linalg.eigvalsh(overlapmatrix)
            print("Smallest Overlap matrix eigenvalues", overlapmatrix_eigvals[:3])
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
        fock_act_on_new_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
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
                mask = np.arange(X.shape[1]) != i #Remove the i-th Gaussian
                X_masked=X[:,mask]
                X_dag_masked=X_masked.conj().T
                XTX_masked=X_dag_masked@X_masked
                I_masked=np.eye(XTX_masked.shape[0])
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
        old_action=self.old_action *sqrt_weights
        functions,minus_half_laplacians=setupfunctions(nonlin_params.reshape((-1,4)),points)
        functions=functions.T
        functions2= functions* sqrt_weights.reshape(-1, 1)
        ovlp_matrix=np.conj(functions2.T)@functions2
        if orbital_norms is None:
            orbital_norms=np.ones(old_action.shape[0])
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
    initial_mask_error,gradient_mask,lincoeff_linear=error_and_deriv(nonlin_params_old[4*nfrozen:],True)
    error_due_to_mask=initial_mask_error
    if initial_mask_error>1e-10:
        solution,new_rothe_error,time,nit=minimize_transformed_bonds(error_and_deriv,
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
    if (nit>=1 or error_due_to_mask>1e-10) and grid_b<grid_b_cancel:
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
    def __init__(self,params_initial,lincoeffs_initial,pulse,timestep,points,nfrozen=0,t=0,norms=None,params_previous=None,method="HF"):
        self.nbasis=lincoeffs_initial.shape[0]
        self.norbs=lincoeffs_initial.shape[1]
        self.method=method
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
        initial_lincoeffs=self.lincoeffs
        initial_params=self.params
        self.time_dependent_potential=self.pulse(t+self.dt/2)*points
        initial_full_new_params=initial_params[4*self.nfrozen:]#+1e-6*(np.random.randn(len(initial_params[4*self.nfrozen:]))-0.5)
        rothe_evaluator=Rothe_evaluator(initial_params,initial_lincoeffs,self.time_dependent_potential,dt,self.nfrozen,self.method)
        initial_rothe_error,grad0=rothe_evaluator.rothe_plus_gradient(initial_full_new_params,printing=True)
        print("Initial Rothe error: %e"%sqrt(initial_rothe_error))
        #sys.exit(0)
        start_params=initial_full_new_params
        ls=np.linspace(0,1,11)
        ls=[0,0.5,0.9,1,1.1]
        best=0
        if self.adjustment is not None:
            updated_res=[initial_rothe_error]
            optimal_linparams=[rothe_evaluator.optimal_lincoeff]
            dx=self.adjustment
            #dx[0::4]=0
            #dx[1::4]=0
            #dx[2::4]=0
            #dx[3::4]=0
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
            initial_rothe_error=updated_res[best]
            optimal_linparam=optimal_linparams[best]
            print("Old Rothe error, using change of %.1f: %e"%(ls[best],sqrt(initial_rothe_error)))
        else:
            print("Old Rothe error: %e"%sqrt(initial_rothe_error))
            optimal_linparam=rothe_evaluator.optimal_lincoeff
        optimization_function=rothe_evaluator.rothe_plus_gradient
        gtol=1e-12
        if optimize_untransformed:
            hess_inv0=np.diag(1/abs(grad0+self.lambda_grad0*np.array(len(grad0))))
            sol=minimize(optimization_function,
                         start_params,jac=True,
                         method='BFGS',
                         options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':gtol})
            solution=sol.x
            niter=sol.nit
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            print("Number of iterations: ",sol.nit)
            new_rothe_error,newgrad=optimization_function(solution)
            print("Rothe Error after optimization: %e using lambd=%.1e"%(sqrt(new_rothe_error),self.lambda_grad0))
            print(list(sol.x))
        else:
            if last_added and self.t>0.5:
                intervene=False
                multi_bonds=0.5
            else:
                intervene=True
                multi_bonds=0.5
           
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
                                                        start_params=start_params,
                                                        gradient=True,
                                                        maxiter=maxiter,
                                                        gtol=gtol,
                                                        both=True,
                                                        multi_bonds=multi_bonds,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=intervene,
                                                        write_file=True,
                                                        timepoint=self.t)
            
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            new_rothe_error,gradient=optimization_function(solution,calculate_overlap=True)
            print("RE after opt: %.2e/%.2e, Ngauss=%d, time=%.1f, niter=%d/%d"%(sqrt(new_rothe_error),rothe_epsilon_per_timestep,len(solution)//4,time,niter,maxiter))
        sqrt_RE=sqrt(new_rothe_error)
        if (initial_rothe_error*1.01<=new_rothe_error and niter>0) or sqrt_RE>rothe_epsilon_per_timestep:
            print("Rothe error increased; OR bigger than acceptable, something needs to be done")
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
                                                        start_params=initial_full_new_params,
                                                        gradient=True,
                                                        maxiter=maxiter,
                                                        gtol=gtol,
                                                        both=True,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=False)
            
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            print("RE after opt: %.2e/%.2e, Ngauss=%d, time=%.1f, niter=%d/%d"%(sqrt(new_rothe_error),rothe_epsilon_per_timestep,len(solution)//4,time,niter,maxiter))
            sqrt_RE=sqrt(new_rothe_error)
        
        smallest_re_onerem=1e10
        if abs(self.t-int(self.t))<=1e-5 or sqrt_RE>rothe_epsilon_per_timestep:
            errors_oneremoved=rothe_evaluator.rothe_error_oneremoved(solution)
            small_i=np.argmin(errors_oneremoved)
            smallest_re_onerem=np.min(errors_oneremoved)
        if sqrt_RE>rothe_epsilon_per_timestep or (smallest_re_onerem<new_rothe_error*1.1 and len(solution)//4>n_extra):
            print("We have to add more Gaussians, Rothe error is too big. But first, let's remove the worst Gaussians, if necessary")
            self.last_added_t=self.t
            print(np.sort(errors_oneremoved))
            small_i=np.argmin(errors_oneremoved)
            if errors_oneremoved[small_i]<new_rothe_error*1.4:
                print("Removing Gaussian %d with error %.3e"%(small_i,sqrt(errors_oneremoved[small_i])))
                solution_removed=np.delete(solution,[small_i*4,small_i*4+1,small_i*4+2,small_i*4+3])
                self.nbasis-=1
                rothe_evaluator.nbasis=self.nbasis
                solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
                                                        start_params=solution_removed,
                                                        gradient=True,
                                                        maxiter=maxiter,
                                                        gtol=gtol,
                                                        both=True,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=False)
                new_lincoeff=rothe_evaluator.optimal_lincoeff
            
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
            err,grad=opt_func_temp(new_params_list[best][-4:])
            hess_inv0=np.diag(1/abs(grad+self.lambda_grad0*np.array(len(grad))))
            sol=minimize(opt_func_temp,new_params_list[best][-4:],jac=True,method='BFGS',options={'maxiter':10,'gtol':gtol,"hess_inv0":hess_inv0})
            very_best=sol.x
            new_rothe_error,gradxx=opt_func_temp(very_best)
            print("Rothe error after first optimization: %e"%sqrt(new_rothe_error))
            best_new_params=np.concatenate((new_params_list[best][:-4],very_best))
            solution_temp=best_new_params
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
                                                        start_params=solution_temp,
                                                        gradient=True,
                                                        maxiter=500,
                                                        gtol=gtol*1e-1,
                                                        both=True,
                                                        multi_bonds=1,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=False)
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            print("Rothe error after optimization: %e"%sqrt(new_rothe_error))
        self.last_rothe_error=sqrt_RE
        new_params=np.concatenate((initial_params[:4*self.nfrozen],solution))
        
        #After opatimization: Make sure orbitals are orthonormal, and apply mask
        new_lincoeff=rothe_evaluator.orthonormalize_orbitals(new_params,new_lincoeff,self.norms)
        if optimize_untransformed:
            print("New Lincoeff:")
            print(list(new_lincoeff))
        new_params,new_lincoeff,self.norms=apply_mask(new_params,new_lincoeff,self.nbasis,self.nfrozen)
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
        print("Avals min and max: %.3e, %.3e; Norms"%(np.min(avals),np.max(avals)),self.norms)
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
                self.propagate(self.t,maxiter*3,last_added=True) #If a gaussian was recently added, we need to optimize more
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
        plt.savefig("Oribtals_t=%.4f.png"%(t), dpi=200)
        plt.close()


initlen=int(sys.argv[1])
n_extra=int(sys.argv[2])
F_input = float(sys.argv[3])# Maximum field strength in 10^14 W/cm^2
maxiter=int(sys.argv[4])
start_time=float(sys.argv[5])
molecule=sys.argv[6]
freeze_start=sys.argv[7]
rothe_epsilon=float(sys.argv[8])
optimize_untransformed=sys.argv[9] #Should be False unless we optimize initial state with imaginary time propagation
if sys.argv[9] != "True":
    optimize_untransformed=False
method=sys.argv[10]
if freeze_start=="freeze":
    nfrozen=initlen
else:
    nfrozen=0

inner_grid=17
if F_input==1:
    grid_b=100
    grid_b_cancel=100
elif F_input==4:
    grid_b=100
    grid_b_cancel=200
elif F_input==8:
    grid_b=200
    grid_b_cancel=400
elif F_input==0:
    grid_b=30
    grid_b_cancel=30
grid_a=-grid_b
muvals_max=grid_b-10
muvals_min=grid_a+10
avals_min=1e-1
avals_max=2
bvals_min=-20
bvals_max=20
pvals_min=-20
pvals_max=20
minimize_transformed_bonds=make_minimizer_function(avals_min,avals_max,bvals_min,bvals_max,pvals_min,pvals_max,muvals_min,muvals_max)
points_inner,weights_inner=gaussian_quadrature(-inner_grid,inner_grid,14*inner_grid+1)
points_outer1,weights_outer1=trapezoidal_quadrature(grid_a, -inner_grid, int(2.5*(grid_b-inner_grid)))
points_outer2,weights_outer2=trapezoidal_quadrature(inner_grid, grid_b, int(2.5*(grid_b-inner_grid)))
points=np.concatenate((points_outer1,points_inner,points_outer2))
weights=np.concatenate((weights_outer1,weights_inner,weights_outer2))
n=len(points)
lambd=1e-9
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
    gaussian_nonlincoeffs=[np.float64(0.7231507218133404), np.float64(0.059902921006840214), np.float64(-0.9415944610876901), np.float64(-1.3411005460174292), np.float64(0.353870990117123), np.float64(0.0171631191065707), np.float64(-0.306654823424573), np.float64(-1.2883057964446927), np.float64(0.9195262347974519), np.float64(-0.3494272008350908), np.float64(-0.12053995700953515), np.float64(-1.171546019686284), np.float64(1.710591853349989), np.float64(0.2820761034226308), np.float64(-0.9088688830279062), np.float64(-0.6729610117264709), np.float64(0.2617858494581348), np.float64(-0.0033493828156299023), np.float64(-0.22236900410397167), np.float64(-0.34171924104847123), np.float64(0.45840667708744953), np.float64(-0.045587106843244456), np.float64(-0.3338283727651104), np.float64(-0.04068555158976227), np.float64(0.43623903395717406), np.float64(-0.004993725118548516), np.float64(0.016810203231239374), np.float64(-0.4645948957169731), np.float64(2.232083616663073), np.float64(2.480003018165945), np.float64(-0.25130389548001814), np.float64(-0.5321511899547341), np.float64(0.5492517934776403), np.float64(-0.145854632750659), np.float64(-0.35388398149670425), np.float64(-0.1228239437238629), np.float64(0.21016571855509378), np.float64(-0.0023784290905808922), np.float64(-0.11784070691450792), np.float64(-0.19994875296433862), np.float64(2.220990419941123), np.float64(-0.5088795282417444), np.float64(-1.3019602679943691), np.float64(-0.3687683881309478), np.float64(0.8381431967762), np.float64(0.7762019618605627), np.float64(-2.9619896701825654), np.float64(-3.8925702060303777), np.float64(1.460869152617643), np.float64(-1.2202315649056314), np.float64(-2.3840955820941137), np.float64(0.08179558804027279), np.float64(0.316770345180818), np.float64(0.0019256045387535331), np.float64(-0.0546754412862857), np.float64(-0.36626945961125607), np.float64(1.879689948395995), np.float64(1.2367228938097574), np.float64(0.2591714736707775), np.float64(0.10300753202932966), np.float64(0.8911656901367777), np.float64(-0.333877829664952), np.float64(-0.9485107603986304), np.float64(-0.5875120759566786), np.float64(0.7178439598534962), np.float64(-0.1589888385084886), np.float64(-0.3666429135978701), np.float64(0.31301095024673237), np.float64(1.865588214115887), np.float64(-2.017717928830478), np.float64(-2.0426467134577293), np.float64(0.13597215787790312), np.float64(1.36716692880775), np.float64(0.7686373076057093), np.float64(-2.090526860963588), np.float64(-0.820278984695464), np.float64(1.1180634209391582), np.float64(-0.3324470962852928), np.float64(-0.7465886728908564), np.float64(-0.005549085999130554)]

    lincoeff_initial=[array([0.036315+0.193935j, 0.559903+0.214208j]), array([0.071565+0.341302j, 0.225306+0.067502j]), array([-0.045469+0.01748j , -0.431271+0.640296j]), array([-0.011207+0.028832j, -0.148858-0.245809j]), array([0.099128+0.042731j, 0.054892-0.045979j]), array([0.520914-0.323844j, 0.069187-0.438157j]), array([-0.385292-0.034012j, -0.221054+0.496293j]), array([-0.00019 -0.000877j,  0.005537+0.00251j ]), array([ 0.099431-0.24809j , -0.143808-0.124601j]), array([0.019645+0.001371j, 0.006691-0.01072j ]), array([0.034983+0.0401j  , 0.040325-0.031827j]), array([0.000012+0.000008j, 0.000385+0.000095j]), array([-0.019119-0.000839j,  0.017677+0.034068j]), array([ 0.191608-0.326026j, -0.138544-0.194675j]), array([ 0.030386-0.004934j, -0.000387-0.037115j]), array([ 0.003709+0.019527j, -0.054148+0.515711j]), array([ 0.204552-0.102202j, -0.042995-0.148849j]), array([0.035177+0.007305j, 0.012135-0.021851j]), array([ 0.021223+0.003723j, -0.008702+0.172325j]), array([-0.344615+0.072408j, -0.161908+0.182556j])]

elif initlen==34 and molecule=="LiH2" and method=="HF":
    gaussian_nonlincoeffs=[np.float64(2.3342162995737774), np.float64(2.440435260296906), np.float64(-0.7172223997059545), np.float64(-3.4932537044451735), np.float64(2.444717377016404), np.float64(0.5077799039043871), np.float64(-0.2269872678256444), np.float64(1.847512890935181), np.float64(0.7299444513835611), np.float64(-0.4346921043659953), np.float64(-0.11058430770879563), np.float64(-1.7532387032149408), np.float64(0.9818853444280399), np.float64(-0.29978827490397586), np.float64(-0.5234838857326124), np.float64(2.7119130864060943), np.float64(0.7027544031234809), np.float64(0.15796724553342364), np.float64(3.719317810811324), np.float64(2.2278826525908677), np.float64(1.7619122591758969), np.float64(-1.4529994577994567), np.float64(-2.70950291767941), np.float64(-2.4112148031960476), np.float64(2.3090358715648907), np.float64(0.11521052909384824), np.float64(-2.126038651507479), np.float64(2.340867278955213), np.float64(2.358966577301044), np.float64(-0.714033822445302), np.float64(-1.7158847962588941), np.float64(-3.4433461230523053), np.float64(0.6000892907880256), np.float64(0.04631545176827453), np.float64(-0.7826758866137413), np.float64(1.8683878104594247), np.float64(1.5783877069189067), np.float64(1.3087233215201717), np.float64(-2.5404604046646035), np.float64(-3.501747729692942), np.float64(2.1225017927907746), np.float64(-1.5538234346050925), np.float64(-0.8785710209250945), np.float64(-2.3197712977410907), np.float64(0.989155026162335), np.float64(-0.8025919480904541), np.float64(-0.7559770499040468), np.float64(-1.988245977872058), np.float64(1.8799238893757444), np.float64(1.9691458998110134), np.float64(-2.1165513468827917), np.float64(1.7946257974046638), np.float64(0.8244050901974749), np.float64(0.7800537638943009), np.float64(-1.2686543636502583), np.float64(1.1189477978108715), np.float64(1.5594868069726748), np.float64(-1.9569776507969077), np.float64(-0.7607934412105828), np.float64(2.997849815033667), np.float64(1.1036831954539046), np.float64(-0.17659764765552827), np.float64(0.06265162908940014), np.float64(-3.1626525577468496), np.float64(1.0242248217082504), np.float64(-0.5144393009273946), np.float64(1.3138069316313417), np.float64(1.0291046353519935), np.float64(1.5513264277619), np.float64(0.9972796682091196), np.float64(-0.6965157134735864), np.float64(-3.1529559375584184), np.float64(0.761043442827057), np.float64(-0.21715275529719602), np.float64(0.9728425769235246), np.float64(-2.948475559819704), np.float64(0.6516651142314743), np.float64(-0.8815185878373031), np.float64(1.0407674791505197), np.float64(0.562310801765485), np.float64(1.3558840162389383), np.float64(1.216644446981756), np.float64(-0.4967239367315011), np.float64(2.6770062673293356), np.float64(0.6674193294775064), np.float64(0.1334399771774932), np.float64(0.1606218933618083), np.float64(1.9878225248531802), np.float64(0.3253845748902293), np.float64(-0.026020195955366573), np.float64(0.5728179286038165), np.float64(-2.2325423887222886), np.float64(0.8506699577832069), np.float64(0.2182822256801552), np.float64(1.334752878929556), np.float64(-1.903638838776107), np.float64(0.33910611875904706), np.float64(-0.022386131844973206), np.float64(-0.37150906270591033), np.float64(1.8506883488788675), np.float64(0.47072546393810716), np.float64(-0.03340087972204187), np.float64(0.3581075262800034), np.float64(-3.108190946385993), np.float64(0.6224833816310713), np.float64(-0.20622663058120716), np.float64(-0.7855188600265243), np.float64(3.7513392954093443), np.float64(0.19722333201494432), np.float64(-0.0008699555247366183), np.float64(-0.09299734360386923), np.float64(0.5177792378755276), np.float64(0.23009947234919154), np.float64(-0.010339831129419497), np.float64(0.24595704355999043), np.float64(-2.4855873846623724), np.float64(0.613704807823036), np.float64(-0.00014605142693988483), np.float64(-0.03941812050073433), np.float64(-3.1236439492201624), np.float64(0.25469585350562884), np.float64(-0.08950737776467924), np.float64(1.5404879556141542), np.float64(-4.431522481013551), np.float64(0.222041628499968), np.float64(0.008616519439987267), np.float64(-0.14024325389319245), np.float64(-1.8803156130734102), np.float64(0.3629325267556285), np.float64(0.055878767369442826), np.float64(2.417522245111262), np.float64(2.0186158149386837), np.float64(0.2572668883891089), np.float64(0.023454085020701648), np.float64(0.24935379706315539), np.float64(0.7334242249213164)]
    lincoeff_initial=[array([-0.00195 +0.000122j, -0.003892-0.000079j,  0.001046-0.003575j, -0.001043-0.00503j ]), array([-0.001649-0.002j   ,  0.001066+0.005702j,  0.002717+0.000253j,  0.002261-0.000374j]), array([-0.164882+0.31909j , -0.192844+0.089521j, -0.170543-0.04163j ,  0.042301-0.06336j ]), array([ 0.051188+0.272465j,  0.09311 -0.046131j,  0.404287-0.195307j, -0.099954+0.161999j]), array([0.000927-0.00109j , 0.00351 +0.003624j, 0.003093-0.003749j, 0.002407-0.000228j]), array([-0.007747+0.007716j, -0.021883-0.003275j,  0.009027-0.013428j,  0.00597 -0.030189j]), array([ 0.001033+0.003997j, -0.002162-0.007969j, -0.000395+0.001428j, -0.005095+0.00136j ]), array([-0.003226+0.000415j, -0.002623-0.002438j,  0.001333-0.001329j,  0.003258-0.002151j]), array([ 0.243542-0.23645j ,  0.251782-0.271739j, -0.654402-0.47096j ,  0.233677-0.21705j ]), array([ 0.015067-0.007341j,  0.019962+0.026025j, -0.022108-0.00043j , -0.04398 +0.025379j]), array([-0.00071 -0.004202j, -0.00033 -0.001517j,  0.002489-0.001908j, -0.001154-0.001179j]), array([-0.087561+0.071165j, -0.084529+0.048292j, -0.071084-0.059549j, -0.053136-0.03757j ]), array([-0.002864+0.00737j ,  0.003909-0.013664j, -0.000541+0.003172j, -0.009624-0.002566j]), array([-0.014639+0.054342j, -0.014499+0.080795j,  0.156652+0.016894j, -0.008268+0.057498j]), array([ 0.004623-0.009683j, -0.003303-0.003879j, -0.019643-0.000628j,  0.004998-0.004714j]), array([-0.00068 +0.865353j, -0.446794+0.42045j , -0.313519-0.009117j, -0.039571-0.452888j]), array([ 0.029146+0.017123j, -0.001303-0.100662j, -0.075703-0.016264j, -0.028386-0.008549j]), array([-0.043572+0.00673j , -0.110461+0.045211j, -0.001883-0.123735j, -0.106053-0.140995j]), array([-0.182892-0.294057j,  0.424099+0.115826j, -0.399454+0.020121j, -0.429232+0.841841j]), array([-0.008607+0.006246j,  0.001218+0.003787j,  0.016256+0.006592j, -0.006235+0.002379j]), array([-0.008318-0.023522j, -0.010638-0.018601j, -0.051966+0.027315j, -0.0038  -0.021079j]), array([-0.344198+0.138844j,  0.469766+0.868105j,  0.823714-0.126314j,  0.203716-0.005108j]), array([ 0.541669+0.426622j,  0.123214+0.241003j, -0.30479 +0.60487j ,  0.111221-0.140868j]), array([-0.306312+0.196765j, -0.00873 -0.095842j, -0.19006 +0.107595j,  0.25406 +0.296939j]), array([ 0.004188-0.364589j, -0.130585-0.140139j, -0.695577+0.176699j,  0.100461-0.239324j]), array([-0.029376-0.550205j,  0.26449 -0.283578j,  0.139494-0.003263j,  0.072449+0.304835j]), array([-0.051438-0.112453j, -0.053007-0.007901j, -0.155892+0.13655j ,  0.000983-0.079496j]), array([-0.017861-0.065986j, -0.033619-0.040098j, -0.166878+0.016308j,  0.016175-0.042221j]), array([ 0.093801-0.083487j,  0.051498-0.046138j, -0.009668+0.005293j,  0.046142-0.056591j]), array([-0.490147-0.190315j, -0.111089-0.615072j,  0.238679+0.01964j ,  0.7678  +0.097305j]), array([-0.003181-0.005314j, -0.000614-0.003139j, -0.003577-0.009365j,  0.000453+0.001952j]), array([0.156767-0.015285j, 0.102329+0.011995j, 0.105004-0.05489j , 0.054939+0.017958j]), array([-0.004706-0.010056j, -0.004403-0.000683j, -0.014494+0.012309j,  0.000323-0.007491j]), array([-0.281488+0.112264j, -0.112743+0.224199j,  0.419559+0.798352j, -0.24351 -0.032635j])]
elif initlen==20 and molecule=="LiH" and method=="DFT":
    gaussian_nonlincoeffs=[np.float64(0.7712055530391919), np.float64(0.021662110026093227), np.float64(-0.999973139306693), np.float64(-1.1916945426203316), np.float64(0.34222161314163146), np.float64(0.015299889492461595), np.float64(-0.33723070911003467), np.float64(-0.8262790172411519), np.float64(0.9855896069152449), np.float64(-0.4624663902472626), np.float64(-0.28930107869557614), np.float64(-1.1602873594984615), np.float64(1.692494735517362), np.float64(0.08072339102237235), np.float64(-0.810775771452055), np.float64(-0.6739733958368205), np.float64(0.25190565166764195), np.float64(-0.0011336848343891997), np.float64(-0.22531201705420603), np.float64(-0.17595598329122702), np.float64(0.43792751946923647), np.float64(-0.032452471343948304), np.float64(-0.27109941420555034), np.float64(-0.05471620103282059), np.float64(0.43246241463328), np.float64(0.008953580600126259), np.float64(0.05660867866346441), np.float64(-0.04762675115729916), np.float64(2.2071095599536084), np.float64(2.19884037498737), np.float64(-0.29768165211214537), np.float64(-0.5224013851047784), np.float64(0.5725300503418292), np.float64(-0.016561761025907777), np.float64(0.10601381390537644), np.float64(-0.004028796247260495), np.float64(0.20879033682178139), np.float64(-0.00045893229072934484), np.float64(-0.11161716655687964), np.float64(-0.178032789828842), np.float64(2.2277620994195306), np.float64(-0.5248109964272709), np.float64(-1.232985108301992), np.float64(-0.2846906431147344), np.float64(0.6482830353765432), np.float64(0.34507075492718636), np.float64(-2.21980064594032), np.float64(-2.5791851864953643), np.float64(1.29292771879431), np.float64(-1.1612104559952094), np.float64(-3.083774388118407), np.float64(0.206269773398925), np.float64(0.3299233628588428), np.float64(0.010470953957421415), np.float64(-0.00153385393278183), np.float64(0.578175759051826), np.float64(1.989527238374532), np.float64(1.2372868469241252), np.float64(0.2505054668395629), np.float64(0.2557527016733582), np.float64(0.8547589555032664), np.float64(-0.35371809097559653), np.float64(-1.3461239349237726), np.float64(-0.7183801075125917), np.float64(0.7095241836837048), np.float64(-0.24469513589252745), np.float64(-0.8093792570199688), np.float64(-0.05358547549971578), np.float64(1.9233774156447285), np.float64(-2.1756322459003337), np.float64(-1.7432336602156402), np.float64(0.1498276554567973), np.float64(1.3542440245237748), np.float64(0.5952682039177254), np.float64(-2.1990692348430674), np.float64(-0.774638780791517), np.float64(1.19464775328089), np.float64(-0.1418235495109314), np.float64(-0.28719628517985724), np.float64(0.10433440833940694)]
    lincoeff_initial=[array([-0.665028-0.598782j,  0.230741-0.239937j]), array([-0.1833  -0.211436j, -0.073691+0.014083j]), array([0.531455-0.23039j , 0.358979+0.410136j]), array([-0.04107 +0.200463j, -0.223851+0.044667j]), array([-0.073126-0.030427j, -0.022266+0.018532j]), array([-0.754102+0.044612j, -0.029119+0.020117j]), array([0.672395+0.085389j, 0.291454+0.064814j]), array([-0.00337 -0.004992j,  0.004524-0.00566j ]), array([0.176114+0.691335j, 0.264411+0.139103j]), array([-0.018476-0.006233j, -0.004749+0.005231j]), array([-0.018993+0.037013j, -0.005731+0.021658j]), array([-0.023342-0.007052j,  0.0013  -0.005285j]), array([-0.009987-0.004399j, -0.001409-0.013172j]), array([-0.085265+0.172782j,  0.049428+0.05708j ]), array([-0.011219+0.016291j, -0.000913+0.007027j]), array([0.371251-0.008381j, 0.10235 +0.324006j]), array([-0.264584-0.17107j , -0.12541 +0.071432j]), array([-0.01725 +0.012442j,  0.000391+0.01024j ]), array([0.069568-0.071398j, 0.135687+0.019787j]), array([ 0.237271-0.223856j, -0.018389-0.060129j])]
    gaussian_nonlincoeffs=[np.float64(0.8639657851711251), np.float64(0.43018698522577914), np.float64(-1.8244132858594582), np.float64(-1.1474354840707135), np.float64(0.3868746639813264), np.float64(0.04572740693197834), np.float64(-0.348958947791873), np.float64(-1.2974517407648702), np.float64(1.2826298637441533), np.float64(-0.5646724397781596), np.float64(-0.8551592284253978), np.float64(-1.447275864859984), np.float64(2.12842702673696), np.float64(-0.6088033556333917), np.float64(0.6356850879289228), np.float64(-1.0097789836077256), np.float64(0.2643048290731481), np.float64(0.0007111495611960719), np.float64(-0.243666086566691), np.float64(-0.2135918475731948), np.float64(0.4834001377858787), np.float64(0.10313250460226124), np.float64(-0.5369779011750871), np.float64(-0.6198653359884694), np.float64(0.5411515894318458), np.float64(-0.05693900843010133), np.float64(-0.4379891933232245), np.float64(-0.2968182790266791), np.float64(1.7566108111774201), np.float64(2.188855789743237), np.float64(1.7410716907080008), np.float64(-0.32603401658545245), np.float64(0.5424267314277981), np.float64(0.15645332205084364), np.float64(-0.07761095375412337), np.float64(-0.1323165954446737), np.float64(0.20254627196087732), np.float64(0.0008580237403714744), np.float64(-0.12712838663368475), np.float64(-0.31820921132092894), np.float64(2.1709161412626967), np.float64(1.1211256159940648), np.float64(0.5297420323488171), np.float64(-0.6913840043824936), np.float64(0.6789888637564461), np.float64(0.5163300987371569), np.float64(-2.817146992418957), np.float64(-1.8554207732913), np.float64(1.4430832468184664), np.float64(-1.9612568020453), np.float64(-2.5816845948747926), np.float64(0.5601870623955513), np.float64(0.3448278604319486), np.float64(0.02198540557090906), np.float64(-0.018486320663306362), np.float64(-0.44204590835361357), np.float64(1.5313895164800624), np.float64(2.2696823955644367), np.float64(2.126568994366008), np.float64(-0.1297338624453954), np.float64(1.088351252930694), np.float64(-0.29894441651959014), np.float64(-2.3883935627511756), np.float64(-1.181792527917739), np.float64(0.9730226898634592), np.float64(-0.7886885603453196), np.float64(-2.56859461030399), np.float64(0.24748547560434225), np.float64(1.456429706447412), np.float64(-1.4395841388139201), np.float64(-1.4556013479524297), np.float64(0.30875727381849444), np.float64(1.8068476815513095), np.float64(2.240955387583611), np.float64(-0.7616891743165883), np.float64(-1.2110958877718547), np.float64(0.8774491308273186), np.float64(-0.05364294816369942), np.float64(-3.888724109550821), np.float64(-0.7046103319632699)]
    lincoeff_initial=[array([-0.020874-1.01613j , -0.283929+0.552845j]), array([ 0.337242-0.188176j, -0.588642+0.280805j]), array([ 0.89925 +0.549393j, -0.293005-0.258323j]), array([ 0.184716+0.143384j, -0.16292 -0.245771j]), array([ 0.011282-0.095377j, -0.038262+0.162458j]), array([ 0.642482+0.344907j, -0.367508-0.439635j]), array([-0.520625+0.125754j,  0.975651-0.02223j ]), array([ 0.043089-0.203911j, -0.101933+0.395152j]), array([-0.483749+0.477185j,  0.493698-0.128621j]), array([ 0.003652-0.016288j, -0.009691+0.027847j]), array([ 0.294484-0.30048j , -0.574856+0.327513j]), array([-0.046896-0.018451j,  0.056171+0.060971j]), array([ 0.037538+0.022207j, -0.053992-0.058005j]), array([-0.398683-0.128751j,  0.662158+0.332427j]), array([-0.085602-0.014104j,  0.171161+0.01065j ]), array([-0.281508+1.023156j,  0.47596 -0.43051j ]), array([ 0.111284-0.008519j, -0.205075-0.008454j]), array([ 0.043256+0.06175j , -0.03743 -0.130436j]), array([ 0.01193 -0.000867j, -0.001746+0.002626j]), array([-0.176911+0.353927j,  0.363347-0.208299j])]

elif initlen==34 and molecule=="LiH2" and method=="DFT":
    gaussian_nonlincoeffs=[np.float64(2.3342162995737774), np.float64(2.440435260296906), np.float64(-0.7172223997059545), np.float64(-3.4932537044451735), np.float64(2.444717377016404), np.float64(0.5077799039043871), np.float64(-0.2269872678256444), np.float64(1.847512890935181), np.float64(0.7299444513835611), np.float64(-0.4346921043659953), np.float64(-0.11058430770879563), np.float64(-1.7532387032149408), np.float64(0.9818853444280399), np.float64(-0.29978827490397586), np.float64(-0.5234838857326124), np.float64(2.7119130864060943), np.float64(0.7027544031234809), np.float64(0.15796724553342364), np.float64(3.719317810811324), np.float64(2.2278826525908677), np.float64(1.7619122591758969), np.float64(-1.4529994577994567), np.float64(-2.70950291767941), np.float64(-2.4112148031960476), np.float64(2.3090358715648907), np.float64(0.11521052909384824), np.float64(-2.126038651507479), np.float64(2.340867278955213), np.float64(2.358966577301044), np.float64(-0.714033822445302), np.float64(-1.7158847962588941), np.float64(-3.4433461230523053), np.float64(0.6000892907880256), np.float64(0.04631545176827453), np.float64(-0.7826758866137413), np.float64(1.8683878104594247), np.float64(1.5783877069189067), np.float64(1.3087233215201717), np.float64(-2.5404604046646035), np.float64(-3.501747729692942), np.float64(2.1225017927907746), np.float64(-1.5538234346050925), np.float64(-0.8785710209250945), np.float64(-2.3197712977410907), np.float64(0.989155026162335), np.float64(-0.8025919480904541), np.float64(-0.7559770499040468), np.float64(-1.988245977872058), np.float64(1.8799238893757444), np.float64(1.9691458998110134), np.float64(-2.1165513468827917), np.float64(1.7946257974046638), np.float64(0.8244050901974749), np.float64(0.7800537638943009), np.float64(-1.2686543636502583), np.float64(1.1189477978108715), np.float64(1.5594868069726748), np.float64(-1.9569776507969077), np.float64(-0.7607934412105828), np.float64(2.997849815033667), np.float64(1.1036831954539046), np.float64(-0.17659764765552827), np.float64(0.06265162908940014), np.float64(-3.1626525577468496), np.float64(1.0242248217082504), np.float64(-0.5144393009273946), np.float64(1.3138069316313417), np.float64(1.0291046353519935), np.float64(1.5513264277619), np.float64(0.9972796682091196), np.float64(-0.6965157134735864), np.float64(-3.1529559375584184), np.float64(0.761043442827057), np.float64(-0.21715275529719602), np.float64(0.9728425769235246), np.float64(-2.948475559819704), np.float64(0.6516651142314743), np.float64(-0.8815185878373031), np.float64(1.0407674791505197), np.float64(0.562310801765485), np.float64(1.3558840162389383), np.float64(1.216644446981756), np.float64(-0.4967239367315011), np.float64(2.6770062673293356), np.float64(0.6674193294775064), np.float64(0.1334399771774932), np.float64(0.1606218933618083), np.float64(1.9878225248531802), np.float64(0.3253845748902293), np.float64(-0.026020195955366573), np.float64(0.5728179286038165), np.float64(-2.2325423887222886), np.float64(0.8506699577832069), np.float64(0.2182822256801552), np.float64(1.334752878929556), np.float64(-1.903638838776107), np.float64(0.33910611875904706), np.float64(-0.022386131844973206), np.float64(-0.37150906270591033), np.float64(1.8506883488788675), np.float64(0.47072546393810716), np.float64(-0.03340087972204187), np.float64(0.3581075262800034), np.float64(-3.108190946385993), np.float64(0.6224833816310713), np.float64(-0.20622663058120716), np.float64(-0.7855188600265243), np.float64(3.7513392954093443), np.float64(0.19722333201494432), np.float64(-0.0008699555247366183), np.float64(-0.09299734360386923), np.float64(0.5177792378755276), np.float64(0.23009947234919154), np.float64(-0.010339831129419497), np.float64(0.24595704355999043), np.float64(-2.4855873846623724), np.float64(0.613704807823036), np.float64(-0.00014605142693988483), np.float64(-0.03941812050073433), np.float64(-3.1236439492201624), np.float64(0.25469585350562884), np.float64(-0.08950737776467924), np.float64(1.5404879556141542), np.float64(-4.431522481013551), np.float64(0.222041628499968), np.float64(0.008616519439987267), np.float64(-0.14024325389319245), np.float64(-1.8803156130734102), np.float64(0.3629325267556285), np.float64(0.055878767369442826), np.float64(2.417522245111262), np.float64(2.0186158149386837), np.float64(0.2572668883891089), np.float64(0.023454085020701648), np.float64(0.24935379706315539), np.float64(0.7334242249213164)]
    lincoeff_initial=[array([-0.00195 +0.000122j, -0.003892-0.000079j,  0.001046-0.003575j, -0.001043-0.00503j ]), array([-0.001649-0.002j   ,  0.001066+0.005702j,  0.002717+0.000253j,  0.002261-0.000374j]), array([-0.164882+0.31909j , -0.192844+0.089521j, -0.170543-0.04163j ,  0.042301-0.06336j ]), array([ 0.051188+0.272465j,  0.09311 -0.046131j,  0.404287-0.195307j, -0.099954+0.161999j]), array([0.000927-0.00109j , 0.00351 +0.003624j, 0.003093-0.003749j, 0.002407-0.000228j]), array([-0.007747+0.007716j, -0.021883-0.003275j,  0.009027-0.013428j,  0.00597 -0.030189j]), array([ 0.001033+0.003997j, -0.002162-0.007969j, -0.000395+0.001428j, -0.005095+0.00136j ]), array([-0.003226+0.000415j, -0.002623-0.002438j,  0.001333-0.001329j,  0.003258-0.002151j]), array([ 0.243542-0.23645j ,  0.251782-0.271739j, -0.654402-0.47096j ,  0.233677-0.21705j ]), array([ 0.015067-0.007341j,  0.019962+0.026025j, -0.022108-0.00043j , -0.04398 +0.025379j]), array([-0.00071 -0.004202j, -0.00033 -0.001517j,  0.002489-0.001908j, -0.001154-0.001179j]), array([-0.087561+0.071165j, -0.084529+0.048292j, -0.071084-0.059549j, -0.053136-0.03757j ]), array([-0.002864+0.00737j ,  0.003909-0.013664j, -0.000541+0.003172j, -0.009624-0.002566j]), array([-0.014639+0.054342j, -0.014499+0.080795j,  0.156652+0.016894j, -0.008268+0.057498j]), array([ 0.004623-0.009683j, -0.003303-0.003879j, -0.019643-0.000628j,  0.004998-0.004714j]), array([-0.00068 +0.865353j, -0.446794+0.42045j , -0.313519-0.009117j, -0.039571-0.452888j]), array([ 0.029146+0.017123j, -0.001303-0.100662j, -0.075703-0.016264j, -0.028386-0.008549j]), array([-0.043572+0.00673j , -0.110461+0.045211j, -0.001883-0.123735j, -0.106053-0.140995j]), array([-0.182892-0.294057j,  0.424099+0.115826j, -0.399454+0.020121j, -0.429232+0.841841j]), array([-0.008607+0.006246j,  0.001218+0.003787j,  0.016256+0.006592j, -0.006235+0.002379j]), array([-0.008318-0.023522j, -0.010638-0.018601j, -0.051966+0.027315j, -0.0038  -0.021079j]), array([-0.344198+0.138844j,  0.469766+0.868105j,  0.823714-0.126314j,  0.203716-0.005108j]), array([ 0.541669+0.426622j,  0.123214+0.241003j, -0.30479 +0.60487j ,  0.111221-0.140868j]), array([-0.306312+0.196765j, -0.00873 -0.095842j, -0.19006 +0.107595j,  0.25406 +0.296939j]), array([ 0.004188-0.364589j, -0.130585-0.140139j, -0.695577+0.176699j,  0.100461-0.239324j]), array([-0.029376-0.550205j,  0.26449 -0.283578j,  0.139494-0.003263j,  0.072449+0.304835j]), array([-0.051438-0.112453j, -0.053007-0.007901j, -0.155892+0.13655j ,  0.000983-0.079496j]), array([-0.017861-0.065986j, -0.033619-0.040098j, -0.166878+0.016308j,  0.016175-0.042221j]), array([ 0.093801-0.083487j,  0.051498-0.046138j, -0.009668+0.005293j,  0.046142-0.056591j]), array([-0.490147-0.190315j, -0.111089-0.615072j,  0.238679+0.01964j ,  0.7678  +0.097305j]), array([-0.003181-0.005314j, -0.000614-0.003139j, -0.003577-0.009365j,  0.000453+0.001952j]), array([0.156767-0.015285j, 0.102329+0.011995j, 0.105004-0.05489j , 0.054939+0.017958j]), array([-0.004706-0.010056j, -0.004403-0.000683j, -0.014494+0.012309j,  0.000323-0.007491j]), array([-0.281488+0.112264j, -0.112743+0.224199j,  0.419559+0.798352j, -0.24351 -0.032635j])]

else:
    raise ValueError("initlen not supported")
gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs)
gaussian_nonlincoeffs[::4]=abs(gaussian_nonlincoeffs[::4]) # a-values can be positive or negative, but we want them to be positive

gaussian_nonlincoeffs=list(np.array(gaussian_nonlincoeffs).reshape(-1,4))
if molecule=="LiH":
    a=5
    b=15
elif molecule=="LiH2":
    a=13
    b=25
nextra_half=n_extra//2
pos_list=np.linspace(a,a+2*(nextra_half-1),nextra_half)
pos_list=np.concatenate((-pos_list,pos_list))
if n_extra != 10 and molecule!="LiH":
    for k in range(len(pos_list)):
        params=[1/sqrt(2),0,0,pos_list[k]]
        gaussian_nonlincoeffs.append(params)
        lincoeff_initial.append(np.array([0]*norbs))
elif n_extra==10 and molecule=="LiH" and initlen==20:
    gaussian_nonlincoeffs=[np.float64(0.7231507218133404), np.float64(0.059902921006840214), np.float64(-0.9415944610876901), np.float64(-1.3411005460174292), np.float64(0.353870990117123), np.float64(0.0171631191065707), np.float64(-0.306654823424573), np.float64(-1.2883057964446927), np.float64(0.9195262347974519), np.float64(-0.3494272008350908), np.float64(-0.12053995700953515), np.float64(-1.171546019686284), np.float64(1.710591853349989), np.float64(0.2820761034226308), np.float64(-0.9088688830279062), np.float64(-0.6729610117264709), np.float64(0.2617858494581348), np.float64(-0.0033493828156299023), np.float64(-0.22236900410397167), np.float64(-0.34171924104847123), np.float64(0.45840667708744953), np.float64(-0.045587106843244456), np.float64(-0.3338283727651104), np.float64(-0.04068555158976227), np.float64(0.43623903395717406), np.float64(-0.004993725118548516), np.float64(0.016810203231239374), np.float64(-0.4645948957169731), np.float64(2.232083616663073), np.float64(2.480003018165945), np.float64(-0.25130389548001814), np.float64(-0.5321511899547341), np.float64(0.5492517934776403), np.float64(-0.145854632750659), np.float64(-0.35388398149670425), np.float64(-0.1228239437238629), np.float64(0.21016571855509378), np.float64(-0.0023784290905808922), np.float64(-0.11784070691450792), np.float64(-0.19994875296433862), np.float64(2.220990419941123), np.float64(-0.5088795282417444), np.float64(-1.3019602679943691), np.float64(-0.3687683881309478), np.float64(0.8381431967762), np.float64(0.7762019618605627), np.float64(-2.9619896701825654), np.float64(-3.8925702060303777), np.float64(1.460869152617643), np.float64(-1.2202315649056314), np.float64(-2.3840955820941137), np.float64(0.08179558804027279), np.float64(0.316770345180818), np.float64(0.0019256045387535331), np.float64(-0.0546754412862857), np.float64(-0.36626945961125607), np.float64(1.879689948395995), np.float64(1.2367228938097574), np.float64(0.2591714736707775), np.float64(0.10300753202932966), np.float64(0.8911656901367777), np.float64(-0.333877829664952), np.float64(-0.9485107603986304), np.float64(-0.5875120759566786), np.float64(0.7178439598534962), np.float64(-0.1589888385084886), np.float64(-0.3666429135978701), np.float64(0.31301095024673237), np.float64(1.865588214115887), np.float64(-2.017717928830478), np.float64(-2.0426467134577293), np.float64(0.13597215787790312), np.float64(1.36716692880775), np.float64(0.7686373076057093), np.float64(-2.090526860963588), np.float64(-0.820278984695464), np.float64(1.1180634209391582), np.float64(-0.3324470962852928), np.float64(-0.7465886728908564), np.float64(-0.005549085999130554), np.float64(0.28511849964404734), np.float64(0.7717610684319318), np.float64(-4.6557248953313), np.float64(-6.0333278403856525), np.float64(0.3121476802084244), np.float64(-0.014889045287444502), np.float64(0.32991335486892437), np.float64(-11.790036651133969), np.float64(1.9997462698098276), np.float64(-3.4450631654069106), np.float64(-12.545636386282188), np.float64(1.5946798688022348), np.float64(0.16870744059237974), np.float64(-0.3316809152296223), np.float64(1.170839465042663), np.float64(-6.5711415620383935), np.float64(0.2030084846936239), np.float64(-0.35716876074596504), np.float64(1.6658769249485594), np.float64(-8.41912692014221), np.float64(0.6970773773108199), np.float64(0.15789912834320607), np.float64(-0.043072883499259085), np.float64(2.7942054702709065), np.float64(0.4866274392633244), np.float64(0.9388334806032201), np.float64(6.917623602405719), np.float64(2.9714298558711363), np.float64(0.9978955014901637), np.float64(0.42091855800478317), np.float64(0.3127474164037589), np.float64(8.713510213610324), np.float64(0.20817178164104588), np.float64(0.2542859208419883), np.float64(1.692311372947069), np.float64(9.437672323251181), np.float64(0.8302020978419351), np.float64(0.0052112909144857444), np.float64(-0.17094533755454466), np.float64(9.920736238845013)]
    lincoeff_initial=[array([-0.073669+0.00212j , -0.173102-0.602433j]), array([-0.016509-0.251693j, -0.138768-0.306493j]), array([-0.165211-0.165277j,  0.737046-0.021888j]), array([ 0.103497-0.069997j, -0.110536+0.23603j ]), array([-0.066316-0.04155j , -0.101909-0.017205j]), array([-0.399504+0.211328j, -0.492587+0.355585j]), array([0.178657+0.032041j, 0.626505-0.138689j]), array([-0.000984+0.002685j, -0.000846-0.005373j]), array([-0.100758+0.157332j, -0.015388+0.269046j]), array([-0.01408 -0.003127j, -0.018162+0.002917j]), array([-0.012871-0.030018j, -0.062713-0.021684j]), array([-0.000049+0.00012j , -0.000138-0.000348j]), array([0.005283+0.010415j, 0.02291 -0.034324j]), array([-0.178374+0.211138j, -0.090039+0.339805j]), array([-0.01614-0.001701j, -0.03723+0.025987j]), array([-0.174206-0.046946j,  0.423221-0.240054j]), array([-0.157795+0.035857j, -0.141319+0.173563j]), array([-0.023236-0.009955j, -0.035368+0.004992j]), array([-0.077245-0.014389j,  0.130549-0.083794j]), array([0.271415-0.047944j, 0.325092-0.043739j]), array([0.+0.j, 0.-0.j]), array([0.-0.j      , 0.-0.000001j]), array([ 0.-0.j, -0.-0.j]), array([0.-0.j, 0.-0.j]), array([-0.-0.j, -0.+0.j]), array([-0.000004+0.000003j, -0.000001+0.000004j]), array([0.+0.j, 0.+0.j]), array([0.-0.j      , 0.+0.000001j]), array([0.      -0.j, 0.000001-0.j]), array([ 0.-0.j, -0.-0.j])]
    gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs)
    gaussian_nonlincoeffs=list(np.array(gaussian_nonlincoeffs).reshape(-1,4))

lincoeff_initial=np.array(lincoeff_initial)
gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs)
num_gauss=gaussian_nonlincoeffs.shape[0]
potential_grid=calculate_potential(Z_list,R_list,alpha,points)


onebody_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)
overlap_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)

time_dependent_potential=0.1*points #I. e. 0.1*x - very strong field
run_HF=True if (start_time<0.05 and lincoeff_initial is None) else False #If we are starting from the beginning, we need to run HF to get the initial state
if run_HF:
    E,lincoeff_initial,epsilon=calculate_energy(gaussian_nonlincoeffs,return_all=True)
    x_expectation_t0=calculate_x_expectation(lincoeff_initial,gaussian_nonlincoeffs)
E0=F0 =np.sqrt(F_input/(3.50944758*1e2))  # Maximum field strength

omega = 0.06075  # Laser frequency
t_c = 2 * np.pi / omega  # Optical cycle
n_cycles = 3
dt=0.05
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
                                timestep=dt,points=points,nfrozen=nfrozen,t=tmax,norms=norms_initial,params_previous=gaussian_nonlincoeffs_prev,method=method)

rothepropagator.propagate_nsteps(Tmax,maxiter)
