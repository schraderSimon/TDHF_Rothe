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
# Function to save or append data
def v_ee_coulomb(grids):
    vp = 1 / np.sqrt(grids**2 + 1)
    return vp
def hartree_potential(grid,rho,weights,vee=v_ee_coulomb):
    v_h = np.zeros_like(rho)
    for i, xi in enumerate(grid):
        v_h[i] = np.sum(rho*vee(xi-grid)*weights)
    return v_h



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
    
    def rothe_plus_gradient(self,nonlin_params_unfrozen,hessian=False,printing=False,calculate_overlap=False,include_overlap_correction=True):
        old_action=self.old_action *sqrt_weights
        gradient=np.zeros_like(nonlin_params_unfrozen)
        if include_overlap_correction:
            calculate_overlap=True
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
        if calculate_overlap or include_overlap_correction:
            overlapmatrix=calculate_overlapmatrix(functions,weights)
            if printing:
                overlapmatrix_eigvals=np.linalg.eigvalsh(overlapmatrix)
                ovlp_mindiag=overlapmatrix-np.eye(overlapmatrix.shape[0])
                print("Smallest Overlap matrix eigenvalues",
                       np.array2string(overlapmatrix_eigvals[:3], formatter={'float_kind': lambda x: "%.2e" % x}),
                         "Biggest Overlap matrix element", np.max(abs(ovlp_mindiag)))
        
        if include_overlap_correction: #This is the correction to the gradient that ensures that the Gaussians don't become linearly dependent
            beta=1e-16
            o=overlapmatrix
            overlap_matrix_abssquared=np.abs(overlapmatrix)**2
            o_abs=np.abs(o)
            Ngauss_tot=o.shape[0]
            N_occ=len(nonlin_params_unfrozen)//4
            overlap_error=0
            prefac=beta/(1-0.99**2)
            for i in range(Ngauss_tot):
                for j in range(i):
                    if i==j or (i<self.nfrozen and j<self.nfrozen):
                        continue
                    
                    if o_abs[i,j]>0.99:
                       
                        overlap_error+=prefac*(o_abs[i,j]**2-0.99**2) #This is the penalty term
                        
                        #Next, we have to update the gradient for gaussians i and j by addin the derivative of the penalty term
                        aideriv=sum(aderiv_funcs[i-self.nfrozen]*np.conj(functions[j])*weights)
                        bideriv=sum(bderiv_funcs[i-self.nfrozen]*np.conj(functions[j])*weights)
                        pideriv=sum(pderiv_funcs[i-self.nfrozen]*np.conj(functions[j])*weights)
                        qideriv=sum(qderiv_funcs[i-self.nfrozen]*np.conj(functions[j])*weights)
                        gradient[4*i-4*self.nfrozen]+=2*prefac*np.real(o[j,i]*aideriv)
                        gradient[4*i+1-4*self.nfrozen]+=2*prefac*np.real(o[j,i]*bideriv)
                        gradient[4*i+2-4*self.nfrozen]+=2*prefac*np.real(o[j,i]*pideriv)
                        gradient[4*i+3-4*self.nfrozen]+=2*prefac*np.real(o[j,i]*qideriv)
                        if j>=self.nfrozen:
                            # If the other Gaussian is not frozen, we also have to update its gradient
                            ajderiv=sum(aderiv_funcs[j-self.nfrozen]*np.conj(functions[i])*weights)
                            bjderiv=sum(bderiv_funcs[j-self.nfrozen]*np.conj(functions[i])*weights)
                            pjderiv=sum(pderiv_funcs[j-self.nfrozen]*np.conj(functions[i])*weights)
                            qjderiv=sum(qderiv_funcs[j-self.nfrozen]*np.conj(functions[i])*weights)
                            gradient[4*j-4*self.nfrozen]+=2*prefac*np.real(o[i,j]*ajderiv)
                            gradient[4*j+1-4*self.nfrozen]+=2*prefac*np.real(o[i,j]*bjderiv)
                            gradient[4*j+2-4*self.nfrozen]+=2*prefac*np.real(o[i,j]*pjderiv)
                            gradient[4*j+3-4*self.nfrozen]+=2*prefac*np.real(o[i,j]*qjderiv)

            rothe_error+=overlap_error
            if printing:
                print("Overlap correction: %e"%overlap_error)
        if hessian:
            hessian_val=np.real(2*gradvecs@np.conj(gradvecs).T)
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
    if initial_mask_error>1e-12:
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
    if (nit>=1 or error_due_to_mask>1e-12) and grid_b<grid_b_cancel:
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
        initial_rothe_error,grad0=rothe_evaluator.rothe_plus_gradient(initial_full_new_params,include_overlap_correction=True)
        print("Initial Rothe error: %e"%sqrt(initial_rothe_error))
        #sys.exit(0)
        start_params=initial_full_new_params
        ls=np.linspace(0,1,6)
        
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
            hess_inv0=np.diag(1/(abs(grad0)+self.lambda_grad0))
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
                maxiter_to_use=3*maxiter
                multi_bonds=0.5
            else:
                intervene=True
                multi_bonds=0.1
                maxiter_to_use=maxiter
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
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
            new_rothe_error,gradient=optimization_function(solution,calculate_overlap=True,printing=True)
            backup_re=sqrt(new_rothe_error)
            backup_solution=solution.copy()
            backup_lincoeff=new_lincoeff.copy()
            print("RE after opt: %.3e/%.3e, Ngauss=%d, time=%.1f, niter=%d/%d"%(sqrt(new_rothe_error),rothe_epsilon_per_timestep,len(solution)//4,time,niter,maxiter))
        sqrt_RE=sqrt(new_rothe_error)
        if (initial_rothe_error<0.99*new_rothe_error and niter>0 and self.t>20) or (sqrt_RE>1.05*rothe_epsilon_per_timestep) or (niter==0 and self.t>20):
            print("Rothe error increased; OR bigger than acceptable, something needs to be done")
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
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
        if (abs(self.t-int(self.t))<=1e-5 or sqrt_RE>rothe_epsilon_per_timestep) or True:
            errors_oneremoved=rothe_evaluator.rothe_error_oneremoved(solution)
            small_i=np.argmin(errors_oneremoved)
            smallest_re_onerem=np.min(errors_oneremoved)
            print("Errors upon removal: ", np.array2string(np.sort(np.sqrt(errors_oneremoved)), formatter={'float_kind': lambda x: "%.2e" % x}))
        if sqrt_RE>rothe_epsilon_per_timestep or (smallest_re_onerem<new_rothe_error*1.1 and len(solution)//4>n_extra):
            print("We have to add more Gaussians, Rothe error is too big, OR A Gaussian can be removed/reoptimized that doesn't contribute enough")
            self.last_added_t=self.t
            
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
            sol=minimize(opt_func_temp,new_params_list[best][-4:],jac=True,method='BFGS',options={'maxiter':50,'gtol':gtol,"hess_inv0":hess_inv0})
            very_best=sol.x
            new_rothe_error,gradxx=opt_func_temp(very_best)
            after_first_lin=rothe_evaluator.optimal_lincoeff.copy()
            after_first_RE=sqrt(new_rothe_error)
            print("Rothe error after first optimization: %e"%sqrt(new_rothe_error))
            best_new_params=np.concatenate((new_params_list[best][:-4],very_best))
            solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
                                                        start_params=best_new_params,
                                                        gradient=True,
                                                        maxiter=500,
                                                        gtol=gtol*1e-2,
                                                        both=True,
                                                        multi_bonds=1,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=False)
            new_lincoeff=rothe_evaluator.optimal_lincoeff
            print("Rothe error after optimization: %e"%sqrt(new_rothe_error))
            sqrt_RE=sqrt(new_rothe_error)
            if sqrt_RE>after_first_RE:
                print("Rothe error increased after second optimization. We will try to optimize from the first solution")
                solution,new_rothe_error,time,niter=minimize_transformed_bonds(optimization_function,
                                                        start_params=new_params_list[best],
                                                        gradient=True,
                                                        maxiter=500,
                                                        gtol=gtol*1e-2,
                                                        both=True,
                                                        multi_bonds=1,
                                                        lambda_grad0=self.lambda_grad0,
                                                        intervene=False)
                new_lincoeff=rothe_evaluator.optimal_lincoeff
                print("Rothe error after yet another optimization: %e"%sqrt(new_rothe_error))
            if sqrt_RE>after_first_RE:
                print("Rothe error increased after third optimization. We will use the single-optimized Gaussian only")
                sqrt_RE=after_first_RE
                solution=best_new_params
                new_lincoeff=after_first_lin
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
        bvals=self.params[4*self.nfrozen+1::4]
        pvals=self.params[4*self.nfrozen+2::4]
        muvals=self.params[4*self.nfrozen+3::4]
        print("Avals: [%.3e, %.3e]; Bvals: [%.3e, %.3e]; Pvals: [%.3e, %.3e]; Muvals: [%.3e, %.3e]"%(np.min(avals),np.max(avals),np.min(bvals),np.max(bvals),np.min(pvals),np.max(pvals),np.min(muvals),np.max(muvals)))
        print("Norms: ",self.norms)
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

inner_grid=10
if F_input==1:
    if method=="HF" and molecule=="LiH":
        grid_b=100
        grid_b_cancel=200
    else:
        grid_b=200
        grid_b_cancel=200
elif F_input==4:
    grid_b=400
    grid_b_cancel=400
elif F_input==8:
    grid_b=600
    grid_b_cancel=600
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
weights=np.concatenate((weights_outer1,weights_inner,weights_outer2))
n=len(points)
lambd=5e-10
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
    gaussian_nonlincoeffs=[np.float64(0.7255192834041255), np.float64(0.061840346837023545), np.float64(-0.9917001285278189), np.float64(-1.3337848317257035), np.float64(0.3453357559952532), np.float64(0.02053235112140228), np.float64(-0.36742695314952606), np.float64(-1.0834956643662734), np.float64(0.9183178717298666), np.float64(-0.345881637150427), np.float64(-0.10836551999261863), np.float64(-1.1043038028594296), np.float64(1.67869796070152), np.float64(0.5613987696570707), np.float64(-1.2021466214944618), np.float64(-0.5870945337846958), np.float64(0.25428442873316837), np.float64(-0.0011130406090233238), np.float64(-0.21910181742842674), np.float64(-0.3336908668575728), np.float64(0.44781926666058147), np.float64(-0.0337275371711964), np.float64(-0.36606953152786276), np.float64(-0.0101279789559661), np.float64(0.4290857532039856), np.float64(0.005105994061942575), np.float64(-0.030082518651047635), np.float64(-0.29534409033286524), np.float64(2.1677600908343395), np.float64(2.495930719469501), np.float64(-0.2222122142539699), np.float64(-0.5158399065017174), np.float64(0.5601607023600819), np.float64(-0.0866167359791015), np.float64(-0.46077579469385477), np.float64(-0.17725339749152874), np.float64(0.20905350666585146), np.float64(-0.0020547963715265077), np.float64(-0.11360694553859044), np.float64(-0.19671592399474067), np.float64(2.146655221886879), np.float64(-0.642601635528554), np.float64(-1.3621790218093635), np.float64(-0.4455537118031207), np.float64(0.7618628142123861), np.float64(0.7725124286301011), np.float64(-3.4436019454198274), np.float64(-3.601247231135282), np.float64(1.62780001550545), np.float64(-1.8175217118151996), np.float64(-1.9484282895720868), np.float64(0.11979097739112161), np.float64(0.3138263555656086), np.float64(0.003840615095615195), np.float64(-0.10583946444628774), np.float64(-0.04408621235882136), np.float64(1.7994297941755015), np.float64(1.558394628784331), np.float64(0.06689700550203855), np.float64(0.049600874230313764), np.float64(0.9658160903349914), np.float64(-0.38225501797079414), np.float64(-0.8583482613107051), np.float64(-0.4724235779137788), np.float64(0.8307451762771515), np.float64(0.03195572440571533), np.float64(0.42823676516284775), np.float64(0.6535560942877344), np.float64(1.7599485931813745), np.float64(-2.909854146669686), np.float64(-1.811726507339289), np.float64(0.13948224234988954), np.float64(1.3960025825992157), np.float64(0.8681732067733704), np.float64(-2.2479920268733498), np.float64(-0.7842278635520102), np.float64(1.144476296721698), np.float64(-0.3563187780441363), np.float64(-0.8178087413634162), np.float64(0.02416509156134444)]

    lincoeff_initial=[array([-0.265954-0.476363j,  0.173018-0.400235j]), array([-0.037565-0.196612j, -0.134229-0.245898j]), array([0.767892-0.244104j, 0.294664+0.378361j]), array([ 0.077466+0.328897j, -0.135548+0.014528j]), array([-0.038782-0.033885j, -0.072334-0.028295j]), array([-0.501803+0.120454j, -0.458819+0.280547j]), array([0.400604-0.120956j, 0.444377+0.081084j]), array([-0.004844-0.003082j,  0.001944-0.001292j]), array([-0.260034+0.286139j, -0.372314+0.523038j]), array([-0.011483-0.002343j, -0.017381+0.0034j  ]), array([-0.013735-0.032354j, -0.008569-0.035348j]), array([-0.000623+0.000016j, -0.00011 -0.000355j]), array([-0.014023-0.014137j, -0.024403-0.022484j]), array([-0.160893+0.098099j, -0.179888+0.236584j]), array([-0.012996+0.012611j, -0.020636+0.006831j]), array([0.078874-0.52436j , 0.304553-0.017403j]), array([0.028418+0.012544j, 0.057703+0.032799j]), array([-0.00123 -0.002043j, -0.002835-0.002435j]), array([0.127773-0.127446j, 0.060037+0.026846j]), array([0.323922+0.09584j , 0.290075-0.047884j])]
    gaussian_nonlincoeffs=[np.float64(0.7369315282821801), np.float64(0.060943240217486296), np.float64(-0.9696177241214851), np.float64(-1.2592582961667025), np.float64(0.35718667190182646), np.float64(0.020428064054160706), np.float64(-0.3223321138648322), np.float64(-1.1109308993493665), np.float64(0.957067723034581), np.float64(-0.29109004131968275), np.float64(-0.32034755142081883), np.float64(-1.0161554287230574), np.float64(1.6658844123687666), np.float64(0.9579330308999002), np.float64(-1.5852278448016168), np.float64(-0.4040913874456688), np.float64(0.2582845741578324), np.float64(-0.0013916488910019644), np.float64(-0.22762168761456897), np.float64(-0.2693610710551233), np.float64(0.45961763303381015), np.float64(-0.03826469557754684), np.float64(-0.34479893034801795), np.float64(-0.06366634727433937), np.float64(0.43931932017351266), np.float64(0.004639436092001097), np.float64(0.024695748708846922), np.float64(-0.343650879973623), np.float64(2.397242859434365), np.float64(2.51955263308398), np.float64(-0.11269439938309442), np.float64(-0.5985713818066646), np.float64(0.5770347002721897), np.float64(-0.10130384087379886), np.float64(-0.42701417232737454), np.float64(-0.2342685961771616), np.float64(0.21114423399064677), np.float64(-0.0015183248872974399), np.float64(-0.11626645097557121), np.float64(-0.18072463952271675), np.float64(2.2353325132044173), np.float64(-1.4734164606056073), np.float64(-1.3858661708444189), np.float64(-0.355406555825135), np.float64(0.6582388715838225), np.float64(0.7502227763343051), np.float64(-4.290449685716489), np.float64(-2.7367000835607347), np.float64(1.6575554383752933), np.float64(-1.4259747642022074), np.float64(-2.229493777016462), np.float64(0.15023678612223557), np.float64(0.3228303052011197), np.float64(0.005953941136364994), np.float64(-0.04895292729637613), np.float64(-0.04759368440746367), np.float64(1.8487317163757329), np.float64(1.1479421162756858), np.float64(-0.5905772149140904), np.float64(0.16325182157880322), np.float64(1.0450749487863762), np.float64(-0.450370664241363), np.float64(-1.042002307412309), np.float64(-0.21090785116347083), np.float64(0.9822154044046787), np.float64(-0.05349456053767097), np.float64(0.21300667978739266), np.float64(1.2426561726622698), np.float64(1.9355165285505425), np.float64(-2.5469573673305366), np.float64(-1.6028810976777825), np.float64(0.147243460098322), np.float64(1.4593966657542938), np.float64(1.135216509366876), np.float64(-2.623995054262496), np.float64(-0.6171851480938769), np.float64(1.1919146218937111), np.float64(-0.2664092461249298), np.float64(-0.5812711512455832), np.float64(0.1459248802583415)]

    lincoeff_initial=[array([-0.2950925320020051+0.0557030544004624j,  0.5133852880982396+0.5785671591378152j]), array([0.1136685699418282-0.0043298319047353j, 0.3158756650380747+0.1983317502914557j]), array([ 0.1366706589126782-0.5951790809566253j, -0.7950371531332456+0.5244005816238181j]), array([0.2114095964415541+0.3104241124591035j, 0.0044021187387148-0.3792393012849827j]), array([0.0265218678931399-0.0212166636724208j, 0.1000223367910638-0.0200990169607749j]), array([-0.0923065813241979-0.1723274810381845j,  0.4000403032219623-0.5938019710421287j]), array([-0.07720301026171  -0.0851200705944271j, -0.5362313335434507+0.3431157376294222j]), array([-0.0025898963562976+0.0001930419381076j,  0.0020690892156754+0.0015767598049143j]), array([-0.0298737220474603-0.2172383671346852j,  0.0014147585045106-0.7114876501727779j]), array([0.0041843090765277-0.00682771677575j  , 0.0198962245898795-0.0108794398970367j]), array([-0.0010881487606998+0.0085168099706184j,  0.0158261343487511-0.0015426346577732j]), array([ 0.0012944717549586-0.0010016757681633j, -0.0023158140879679-0.0003783267658676j]), array([0.0096712511754571+0.0386278850970437j, 0.0277424516347078+0.0087865422453223j]), array([-0.0690668310330724-0.1116791705030292j, -0.0419059365840657-0.353953262423501j ]), array([0.0210854821824963+0.0246215116412038j, 0.0012933700973773-0.0618240104114665j]), array([-0.4261478837422948-0.1757540974584176j,  0.2093144644061359+0.5865313954374654j]), array([-0.0032407522542861-0.005125577737997j , -0.0063675042692439-0.0089102818170481j]), array([0.0000282703568964-0.00234280406727j  , 0.0010480067545816-0.0100551739227366j]), array([ 0.1357692452255108-0.0993282638586122j, -0.1874245259268906+0.0394036044555611j]), array([ 0.181632403054413 +0.0509449097776507j, -0.3919187235555229+0.1685375662243869j])]

elif initlen==34 and molecule=="LiH2" and method=="HF":
    gaussian_nonlincoeffs=[np.float64(2.3006401335443254), np.float64(2.419627487988478), np.float64(-0.6892709833789855), np.float64(-3.4755861943182906), np.float64(2.4442867728535096), np.float64(0.35526945711430064), np.float64(-0.1732694073157898), np.float64(1.8453811746303619), np.float64(0.7387517320397643), np.float64(-0.42577350479421905), np.float64(-0.14513683038551822), np.float64(-1.7857493780979286), np.float64(0.9864022191223563), np.float64(-0.30384756265807183), np.float64(-0.5236406518985024), np.float64(2.7146619280976454), np.float64(0.685273735781145), np.float64(0.13861952672822744), np.float64(3.785683224106183), np.float64(2.1727557559570903), np.float64(1.7832286953931105), np.float64(-1.4434199625018724), np.float64(-2.5624805587740997), np.float64(-2.390224797367677), np.float64(2.320720372421454), np.float64(0.011947289940559167), np.float64(-1.9955825972131922), np.float64(2.34612580020303), np.float64(2.333653471547741), np.float64(-0.9706430408400013), np.float64(-1.7162287131486162), np.float64(-3.4266197688145823), np.float64(0.600689456585156), np.float64(0.04688444142364901), np.float64(-0.7964724320271611), np.float64(1.8945898631476232), np.float64(1.5887191110409158), np.float64(1.2717289536934018), np.float64(-2.459046702917216), np.float64(-3.467432372674582), np.float64(2.1276760629823004), np.float64(-1.6144602156624897), np.float64(-0.8566997242342776), np.float64(-2.325034698570429), np.float64(0.990050385609722), np.float64(-0.8229049988971069), np.float64(-0.7998682739182552), np.float64(-2.0100209754541796), np.float64(1.8844601293021304), np.float64(1.9384280240122116), np.float64(-2.1085786822883383), np.float64(1.789109341578251), np.float64(0.8239177474888838), np.float64(0.778767153500414), np.float64(-1.2466348555468727), np.float64(1.1124709223302403), np.float64(1.5631154977785808), np.float64(-1.9705948360627696), np.float64(-0.7268638888740965), np.float64(3.0013047352431794), np.float64(1.1051037390169545), np.float64(-0.16675095498015027), np.float64(0.04314914514638123), np.float64(-3.1556351879363578), np.float64(1.0183100397142006), np.float64(-0.5000110292578452), np.float64(1.3148216147922585), np.float64(1.0413792675666647), np.float64(1.5567806148971295), np.float64(0.9786386956959157), np.float64(-0.6642873939655879), np.float64(-3.1435314856369265), np.float64(0.7646607589450125), np.float64(-0.21994203363391876), np.float64(1.05226957831923), np.float64(-2.949387484585731), np.float64(0.657943602900683), np.float64(-0.8921051348345146), np.float64(1.1212738766721098), np.float64(0.6176273879096035), np.float64(1.3566409799835777), np.float64(1.2054568804715204), np.float64(-0.5060499023271738), np.float64(2.680986656543605), np.float64(0.6660164276425797), np.float64(0.14027150944864014), np.float64(0.15573777333094638), np.float64(1.9902307924423663), np.float64(0.32472467801464816), np.float64(-0.025520092463880022), np.float64(0.5725675310331696), np.float64(-2.260009553659389), np.float64(0.8615991469522906), np.float64(0.21738867015992186), np.float64(1.352315996089856), np.float64(-1.8865622471856913), np.float64(0.34275055346633176), np.float64(-0.022604340038030774), np.float64(-0.36678309709733864), np.float64(1.8600337175398218), np.float64(0.4671874045107948), np.float64(-0.029494875291995806), np.float64(0.3569208883408571), np.float64(-3.0757189932360087), np.float64(0.6305413808052774), np.float64(-0.2124221066068054), np.float64(-0.8107370118747668), np.float64(3.763204523147577), np.float64(0.19656542795643275), np.float64(-0.0008776046594668322), np.float64(-0.09481967588369856), np.float64(0.46674606968676957), np.float64(0.22849978703338125), np.float64(-0.009977237949789785), np.float64(0.24202624913520224), np.float64(-2.3796903825457623), np.float64(0.6105642920762232), np.float64(0.001251047992436387), np.float64(0.009080847607482), np.float64(-3.092712790653501), np.float64(0.25325542117549277), np.float64(-0.08912634037401566), np.float64(1.5577453925230162), np.float64(-4.361940584951661), np.float64(0.22087892381383042), np.float64(0.009366529964754922), np.float64(-0.1578669166180696), np.float64(-1.7917159118051849), np.float64(0.3636232723107457), np.float64(0.05470495914459701), np.float64(2.428530774845887), np.float64(1.9848090819228676), np.float64(0.2568666152670833), np.float64(0.02298533755998491), np.float64(0.2580752277578288), np.float64(0.6839726221711965)]

    lincoeff_initial=[array([-0.0031944857206451+0.0048257854195362j, -0.0028173123318757+0.002524539588385j , -0.0026919869310758+0.0001513290424519j,  0.0032166795416315+0.0023731841479517j]), array([-0.0034022517394194-0.0009310767114645j,  0.0021901943239479-0.0004083288282796j,  0.0046309639927858+0.0009838477261683j,  0.0037729390560648+0.0003334982426869j]), array([-0.083972932096842 +0.08084810505947j  , -0.2185696704944604+0.1491100526305942j, -0.1315309431330566+0.2924979078858719j,  0.0824891907115638-0.300916492232738j ]), array([-0.189511929922747 +0.2445497754927522j,  0.2733376273019569-0.2721441659690797j, -0.1436432357372895+0.0788090870776391j, -0.1747065403723622-0.0621020673379842j]), array([-0.0009139762696156+0.0032845908838371j, -0.0000055862618857-0.0037765683527373j,  0.0037731467315694-0.0020147183050092j,  0.0011273568033686-0.0001410562890213j]), array([-0.0236042975856844+0.0184229218534744j, -0.0179859089175481+0.0121945019653672j, -0.0090463965812976+0.002926099348657j ,  0.0052705238106056+0.0094788805833627j]), array([ 0.002835296772724 +0.0000712104219641j,  0.0002938593887149+0.001684495590985j , -0.0087925251977403-0.0002234917422107j, -0.0056460105503177-0.0000850634617556j]), array([-0.0036483986282267-0.0010267213346841j, -0.0033287824432174-0.001303068044069j , -0.0021972306308619-0.0005675973792102j,  0.0008857972892413+0.000859792663138j ]), array([ 0.5357835932050034+0.2187037475044637j, -0.6975617452284087-0.1652633234696158j,  0.2221064005486789-0.2851826147680241j, -0.0176648718162568+0.0896578381170334j]), array([0.0494906915910231+0.0026971132535383j, 0.0356628853054433-0.0000087158909911j, 0.0107659719022989+0.0015471837983466j, 0.0091016493123717-0.0126090610014049j]), array([-0.0006055245613144+0.000835565410369j ,  0.0006576956209712-0.0009656745168139j, -0.0011586220768396-0.0026452968341111j,  0.0020024119073843+0.003631182018111j ]), array([-0.0016394780309984+0.0762959936316667j, -0.0532863746142034+0.0660010467625858j, -0.0686720356022378+0.0702832339362491j,  0.0955485917234724-0.0603168367413446j]), array([ 0.0026232028507343+0.002297314704643j ,  0.0002074922425284+0.0033548749338207j, -0.0123756143100963-0.0091924612982202j, -0.0076613284492145-0.0061509038716244j]), array([-0.1189023055813204+0.004631045084266j ,  0.1381520274779189-0.0332169039168178j,  0.0011490341932564+0.0718587281069998j, -0.0005222973817727-0.0124170121206005j]), array([ 0.0132132306656305-0.0043440616324248j, -0.0158675384048175+0.0053545640272732j,  0.0044962558391728-0.0023390514992338j,  0.002438049670597 +0.0050832492813328j]), array([-0.195851823983265 +0.4972132899006508j, -0.3161235676959051+0.6991451524916946j,  0.0020112840818442+0.7007341033283231j, -0.0668576670732504-0.5581492679992262j]), array([ 0.0819365012663447+0.015167941189952j , -0.0671757587462011+0.0079630802484926j, -0.0629163186473779-0.0298583300037112j, -0.0559190464947816-0.0012528308173834j]), array([-0.0275769470343716+0.1646211924412198j, -0.03464433645262  +0.0894946318900504j, -0.0672910289695846+0.0150863022680338j,  0.1142201697193262+0.0506705926452924j]), array([ 0.7770097835612594-0.3160020107832421j,  0.4681997088783619-0.3649462888823372j, -0.0243676336006154-0.1106495892068824j,  0.2975249165607736-0.2397009216515817j]), array([-0.0119350203263495-0.0012268764706568j,  0.0144477718652268+0.0009790834846165j, -0.0057309130834527+0.000078728841603j ,  0.0005899521906984-0.0048902415459964j]), array([ 0.0315966458887013-0.0272931271433275j, -0.0352550806673119+0.0382733044830636j, -0.005312052640588 -0.0202584637262866j,  0.0090108327519815+0.0032709783395067j]), array([-0.8082182406184488+0.2386037467035516j,  0.6397640833401641-0.3444323452747445j,  0.5826701832932554+0.0330806083417124j,  0.3881042708102095-0.3470374869379947j]), array([ 0.0663842263785725-0.2099883354248412j, -0.033541787838914 +0.5429843461394023j,  0.4645485038809394+0.2363209825566481j, -0.6181764750248554-0.3237649977882119j]), array([ 0.0556390576551484-0.3218022809606765j, -0.134865415636287 -0.2034575644717815j, -0.1616424000278519+0.1156678031081108j,  0.0988784282686989-0.3233760912527431j]), array([ 0.4380210037929789-0.2704805853580797j, -0.5485056808638215+0.3429659272758762j,  0.103713377567065 -0.1682413829633961j,  0.1530725980888409+0.1359631049147658j]), array([0.177193971017891 -0.3116310650317843j, 0.1631599440987232-0.4199270956741304j, 0.0290929102646065-0.4134105351359151j, 0.0387657671778858+0.339537576228966j ]), array([ 0.0765247375066828-0.1282779201835943j, -0.0987882051725469+0.1562455540937803j,  0.0179229632171724-0.0572859941333008j,  0.0705728000156372+0.0250487015618437j]), array([ 0.1074647870303741-0.0485982136298657j, -0.1313872658903988+0.0599475589258747j, -0.0000866155539637-0.029622533309382j ,  0.0456522992645394+0.0139941390239962j]), array([ 0.0282068129379003-0.0159314192533427j, -0.0526425166105729+0.0236083280978691j,  0.0866385569894758-0.0706788739330372j, -0.0698023952111204+0.0717727313732032j]), array([-0.4854745851708375-0.6202351131886755j, -0.4943725420472897-0.6108414840037755j, -0.3340740899415017-0.2922846511866505j,  0.0662096879119304+0.1864565836684536j]), array([ 0.0052433316579481+0.0039020180742947j, -0.0044171295268181-0.0076948822696239j, -0.0027433600924527-0.0024241883446555j,  0.0064522623825169+0.0047337109541716j]), array([-0.0052553420517669+0.0688516293667958j,  0.0180407708452083-0.1007586189049702j,  0.1135393396357256-0.0150505604424428j, -0.1005476571960437+0.0639637001630009j]), array([ 0.0057435970107704-0.0123705639718795j, -0.0077782772834101+0.0153509032473325j,  0.0013070137860105-0.0056892110816057j,  0.0065441150995673+0.0012975785177348j]), array([-0.5181176052500773-0.4127563733890762j,  0.6473965095635413+0.4930770589967639j, -0.1216250451849095-0.0125643071415425j, -0.0221569968671651-0.1757119500134398j])]


elif initlen==20 and molecule=="LiH" and method=="DFT":
    gaussian_nonlincoeffs=[np.float64(0.8632971637929692), np.float64(0.4322817379795828), np.float64(-1.8003704736095572), np.float64(-1.1472060958741754), np.float64(0.38491378799960857), np.float64(0.046481656661276655), np.float64(-0.35670826289749075), np.float64(-1.2824252711994368), np.float64(1.285625519274064), np.float64(-0.5642042077572141), np.float64(-0.8100338973271123), np.float64(-1.4454010277854907), np.float64(2.1127516041534924), np.float64(-0.6838226371415042), np.float64(0.6852077949662304), np.float64(-0.9997235538670199), np.float64(0.2639789598521965), np.float64(0.001064520027487265), np.float64(-0.24386358151326265), np.float64(-0.2155479584206956), np.float64(0.4822477446134499), np.float64(0.10098318870433846), np.float64(-0.537623394697009), np.float64(-0.6118927162394862), np.float64(0.5428118989629951), np.float64(-0.054714393795667435), np.float64(-0.4410891410920785), np.float64(-0.291799076224867), np.float64(1.7762159651000073), np.float64(2.1450034416548296), np.float64(1.699204942648232), np.float64(-0.30836810092136835), np.float64(0.5418477732220092), np.float64(0.15341550422393485), np.float64(-0.07751898050138407), np.float64(-0.1242243801618327), np.float64(0.20254351045671176), np.float64(0.0008453053441559824), np.float64(-0.12689018540542935), np.float64(-0.3138730941506729), np.float64(2.1756500663373797), np.float64(1.023497632872557), np.float64(0.5035422545508811), np.float64(-0.6772439837499679), np.float64(0.6732957941440587), np.float64(0.5161705886475297), np.float64(-2.8098680870770814), np.float64(-1.8664903935227852), np.float64(1.447695521771797), np.float64(-1.9510638678936454), np.float64(-2.4988557692804343), np.float64(0.5708119484086541), np.float64(0.34404898660168876), np.float64(0.022456380047616974), np.float64(-0.024122388507252263), np.float64(-0.4297891149154676), np.float64(1.5534950887889978), np.float64(2.249678973348158), np.float64(2.0800477461591473), np.float64(-0.11037513104828045), np.float64(1.0867671992574415), np.float64(-0.2932577718804815), np.float64(-2.3880680511592396), np.float64(-1.177989946682788), np.float64(0.9729441229302904), np.float64(-0.7849677405293959), np.float64(-2.519456408953336), np.float64(0.2696933230803403), np.float64(1.4335720042493365), np.float64(-1.3853981730065288), np.float64(-1.2305237149248098), np.float64(0.3152660724763152), np.float64(1.7859449903993738), np.float64(2.157972642066431), np.float64(-0.8826600266372654), np.float64(-1.1931871544659285), np.float64(0.8760464295007526), np.float64(-0.05263254390637101), np.float64(-3.885246875694081), np.float64(-0.6989717693173054)]

    lincoeff_initial=[array([ 0.3259857703461609-0.1317820602192271j, -0.1987927662736797+1.034152943089634j ]), array([-0.0745987482054268+0.3366975509328219j, -0.5128201241840777+0.4284329885483658j]), array([ 0.0116547063027692+0.5620246503509252j, -0.6845893114950392-0.596331908594896j ]), array([-0.1070105516832839+0.0369256610257711j, -0.2582922211582784-0.201951917258872j ]), array([0.0503965859427288+0.0688079290853141j, 0.0047242594557246+0.1709975517723861j]), array([-0.1098437228938248+0.2539553380126321j, -0.7557700710703548-0.4812235021104262j]), array([0.3100803768287482-0.3707632135313202j, 0.9598949031850474-0.3660761731156844j]), array([ 0.0967336916139454+0.1826259862276709j, -0.0021065369752173+0.3617481134785029j]), array([-0.1738589724438909-0.1157929019929642j,  0.6929536442905934-0.4875717778927042j]), array([ 0.0078872258021425+0.0130284772311991j, -0.0013459159434921+0.0303496502122068j]), array([-0.0211035025659611+0.2688948245357189j, -0.4596272618008139+0.5001070153854831j]), array([0.0307276274626404+0.00243945101602j  , 0.0713275465912703+0.0428140190201801j]), array([-0.0352861043902788+0.0028453101727748j, -0.075767253227245 -0.0306087959006949j]), array([0.3421774000642045-0.1314603252422835j, 0.7414520616328643+0.1534212961289984j]), array([0.0771621870647895-0.0583783770317532j, 0.1455302473276784-0.0170662126068737j]), array([-0.3435806325014436+0.0687647057876295j,  0.5055159520331891-0.9587946128816721j]), array([-0.0651090403600993+0.0729075248864031j, -0.1875032343405501+0.0572278394459836j]), array([-0.0496998358086041-0.0215172797276681j, -0.0860603477726349-0.0987823310775429j]), array([ 0.0078938174123421+0.007479886253648j , -0.0116236669108772+0.0013621175165137j]), array([-0.0513848264044742-0.0821334431939806j,  0.3105125535069322-0.4182166714098785j])]

elif initlen==34 and molecule=="LiH2" and method=="DFT":
    gaussian_nonlincoeffs=[np.float64(2.2324274791700396), np.float64(1.5820545154857588), np.float64(-0.9427087016041371), np.float64(-3.8368547834254034), np.float64(2.376354302079408), np.float64(-0.4806384130439742), np.float64(0.943332210548159), np.float64(1.7789862985972436), np.float64(1.4818485716220469), np.float64(-2.1278935025200987), np.float64(-0.18533310565880168), np.float64(-4.118937788392002), np.float64(1.0580098315962552), np.float64(-0.544407114404264), np.float64(-0.3208766692214884), np.float64(2.7136380799943898), np.float64(0.6694081664802133), np.float64(0.22993345397345105), np.float64(3.5484782656520677), np.float64(2.5387705728886516), np.float64(1.218665734781728), np.float64(-1.6915079264759436), np.float64(-1.4207400460726423), np.float64(-1.378320651436759), np.float64(2.1434540933558126), np.float64(1.8363918729309674), np.float64(1.381875047108402), np.float64(2.3325405180674226), np.float64(2.007067295627287), np.float64(-0.7169196800646019), np.float64(-1.7118061479660256), np.float64(-3.4809305964429194), np.float64(0.7822085759910646), np.float64(0.11538298665057753), np.float64(0.08437386689851101), np.float64(1.8878139024410796), np.float64(1.8902506384587896), np.float64(2.083563687330402), np.float64(-1.6493258397674784), np.float64(-4.104271058469698), np.float64(1.9752732752290458), np.float64(-2.1639587547167327), np.float64(-0.8377516130379954), np.float64(-2.361344354888308), np.float64(1.0292831208119546), np.float64(-0.6333375137846528), np.float64(-0.4286298027073693), np.float64(-2.1797847202908036), np.float64(2.17702797659417), np.float64(-1.7205458724139349), np.float64(1.5607108019314697), np.float64(1.519660907156213), np.float64(0.7823983202979665), np.float64(0.6270887026202316), np.float64(-1.3302409899015695), np.float64(1.4605654867381628), np.float64(1.6071049901347114), np.float64(-0.1529269474206193), np.float64(0.00893377298572309), np.float64(3.108398866846258), np.float64(0.9516349014944687), np.float64(-0.34114725700271725), np.float64(0.09914008198316551), np.float64(-3.007819080111033), np.float64(1.32475982714859), np.float64(0.3611304796597194), np.float64(0.4790827215621077), np.float64(0.7350512061733989), np.float64(1.3157625530745298), np.float64(1.1059071902308721), np.float64(-0.7389788638119745), np.float64(-3.3707613809878083), np.float64(0.6895269769211396), np.float64(-0.005451357909719052), np.float64(1.6363429841004484), np.float64(-2.53974410015939), np.float64(0.6499017256400268), np.float64(-0.7077272665613954), np.float64(1.4139074577883008), np.float64(0.5560413829823425), np.float64(1.428804220743978), np.float64(0.7935140484257944), np.float64(-0.24846851132994627), np.float64(3.031201012221721), np.float64(0.7221146157655709), np.float64(-0.028896490611344805), np.float64(-0.7131080145813804), np.float64(2.668425085632484), np.float64(0.4438132824777762), np.float64(-0.009547831832746281), np.float64(0.7863747621538862), np.float64(-2.973865817756038), np.float64(0.6954926564909748), np.float64(0.2911248980785833), np.float64(1.923364330306065), np.float64(-1.9963798369025787), np.float64(0.3889921705590962), np.float64(0.029511723748534283), np.float64(0.3947900467139951), np.float64(2.6019757620896775), np.float64(0.5194433049605354), np.float64(-0.007788616741058774), np.float64(0.5565736053566013), np.float64(-3.3346141227342163), np.float64(0.5796455258349767), np.float64(-0.16215662220846536), np.float64(-0.9683942827436435), np.float64(3.1452339506293003), np.float64(0.16958937977839256), np.float64(-0.0022252328231063523), np.float64(-0.104003556011724), np.float64(0.6311394596331765), np.float64(0.24699720287166885), np.float64(-0.052346495359717374), np.float64(0.8735090282538075), np.float64(-4.049930462695657), np.float64(0.7020402301442439), np.float64(-0.012636316036537901), np.float64(0.3407792779878188), np.float64(-3.3536136174852214), np.float64(0.33509420422649633), np.float64(-0.15172708335619048), np.float64(1.9780367190437678), np.float64(-3.9249190852588165), np.float64(0.22771324778252178), np.float64(-0.008996063495979832), np.float64(0.20944373940465735), np.float64(-2.4044298149993915), np.float64(0.5925166687410821), np.float64(-0.27262258872452605), np.float64(1.1635176565906042), np.float64(0.7591423268752346), np.float64(0.25401033922393923), np.float64(0.02811191789517091), np.float64(0.4415208266873198), np.float64(1.6392255413921926)]

    lincoeff_initial=[array([ 0.0074363127400882+0.0044454337475505j,  0.0203823625913529+0.00195456084619j  ,  0.0096094985614974-0.0126318220204744j, -0.0048934391048265-0.0025329427344564j]), array([ 0.0021147832756563-0.0060409875737753j,  0.0023449518979374-0.005822517019913j , -0.0014128308119944+0.0061383845687851j,  0.0011846805604539-0.0204365736024967j]), array([-0.000913769511981 -0.000645198815108j ,  0.0017201008092675+0.0006001406036634j,  0.0005491372978778-0.0013488591976185j,  0.0002650534197768-0.00040157022453j  ]), array([ 0.0403826989550634+0.1510920700855779j, -0.1542641167305012+0.2277487021979327j, -0.4494642877459087-0.1573327556448379j, -0.1069440093223926-0.2168840123187108j]), array([ 0.0083457322268491-0.0012824929703635j,  0.0118920266459004+0.0169336603635867j, -0.0106418907512111+0.0310383642813219j, -0.032437666984285 +0.0166427841865657j]), array([-0.0087252461844826+0.0006294748927554j,  0.0018373225824399-0.0054758803862083j, -0.0060773331472382-0.0051068743515503j,  0.002628372098416 -0.0027865146736898j]), array([ 0.0008918712074634-0.0002084159452182j,  0.0013271197612472-0.0001634513614895j, -0.0003602896602196-0.0006930388407363j,  0.0043155949269762-0.0001808290100221j]), array([-0.0204304637117912+0.0077137173873554j, -0.0058524223669848-0.0073919027973558j, -0.0140652454983265+0.0050512886928794j,  0.0026366414402246-0.0047993996752486j]), array([-0.0645786035707923-0.365947261320494j ,  0.1417692099204649-0.5989008864713838j,  0.4062931039173011+0.4391426640104573j, -0.0028723978501716-0.7774108302448727j]), array([-0.0050315589480034-0.0059330632052115j,  0.0024202838118172-0.0239902894651282j, -0.0182417594547765-0.0104565092849622j, -0.0010657732715264+0.0027265420311963j]), array([-0.0030737761912785-0.0013957192887855j,  0.0014705221173704-0.0019952856720107j, -0.0020205801932376-0.0011373358592047j,  0.0000748064136538+0.0002272162733324j]), array([-0.1171969813727486+0.3380243014960659j, -0.1411528200425814-0.2405313722632059j, -0.2125916313855677+0.1430875375240618j, -0.0455672962301178-0.0430439474131001j]), array([ 0.0035217516262843+0.0003245680180299j,  0.0022404058331826-0.0000920160630054j, -0.0038239722659599-0.0000517750192044j,  0.0104329801784581-0.0050994335243747j]), array([ 0.0820686265677134+0.0052757233524695j,  0.0194560183034482+0.156139191346995j , -0.3334541601665744+0.1560202008551208j, -0.2309728519344987-0.2451226310713473j]), array([-0.0295873445742505+0.0062732578427279j, -0.026311461618982 -0.0501220889083784j,  0.0891649265790388-0.0874584024961185j,  0.0940175926119246+0.0466097987531512j]), array([ 0.6358577092300156-0.2749783227860205j, -0.7831567715660641+0.1282936622530965j,  0.1282717211964287+0.4717710070175463j,  0.0475740692913805+0.3099098354798153j]), array([ 0.0261253936134846-0.0061905334210191j,  0.0325384107211049+0.0328683662149382j,  0.004304549519536 +0.0766268339862946j, -0.0552310285145706+0.0568213918280993j]), array([-0.0983559256567883+0.0537976254417779j, -0.0035261713983595-0.0700198482893383j, -0.0853555236568765-0.0026453096261697j,  0.0040859889331135-0.0287247042424769j]), array([-0.2559125228381554+0.1001332147643044j, -0.2445285732438194-0.1261711642489024j, -0.2497709857399179+0.2105488394907833j,  0.0300942102133935-0.0177006879821615j]), array([-0.010556733330232 +0.0222914608822112j, -0.0488229912557041-0.0067464566402113j, -0.0134757177843976-0.0984917760795974j,  0.067573679875362 -0.0432063628265801j]), array([-0.0330320133945885-0.0269413813915298j,  0.0400003272955749-0.0718562183785499j,  0.1915483523729954+0.0384210485465398j,  0.0207417864414267+0.1789766180311118j]), array([ 0.1410866452558156-0.0888207116740206j,  0.2753019710337583+0.2498225189832748j, -0.4958888505273332+0.6504209992881859j, -0.5932974408940793-0.1808919845299313j]), array([0.2821786929215188-1.4211846721729784j, 0.3964145944681555+0.4960980052353903j, 0.4912446779675489-0.7319526430597788j, 0.2774674404389015+0.2156428088852117j]), array([0.024038813787284 +0.0007346496092144j, 0.0173645690839513+0.193187719773643j , 0.1552180430448853+0.0455985670631765j, 0.0184752612116907-0.0159631248112028j]), array([-0.127425406307348 -0.0983040119830432j,  0.1663232291063637-0.2759295086135457j,  0.6745382966634902+0.1174497874294177j,  0.122291090559798 +0.6422661551257319j]), array([-0.7236232697461247+0.6698389801846919j,  0.7867735996351555-0.1668525511374954j, -0.2951044550605527-0.3758045936818571j, -0.1405043344286937-0.5066287869436291j]), array([ 0.0487002473552986+0.0455536276085828j, -0.1014616685681688+0.1173260103238521j, -0.3335885181100563-0.2500876292390031j,  0.1351988106905769-0.3538359916937251j]), array([ 0.0089976783178937+0.0043877338561749j,  0.0028015570716526+0.0123931915898237j, -0.0195104862624599+0.0335795523104093j, -0.0328632313801601-0.00709450191522j  ]), array([0.0326092919598576-0.0912107542041128j, 0.0455654046768253+0.0587329612546917j, 0.0716315021667938-0.0686203372343386j, 0.0298730265208758+0.0108880436084257j]), array([ 0.877871147231209 +1.0304965235369201j, -0.0049319255325161-0.3958240968986482j,  0.2078503869581607+0.2651569415353479j, -0.4256297365307248+0.1117261447234154j]), array([ 0.018355959710942 +0.1357705478189696j, -0.0930290011572914-0.0487592759338255j, -0.0354660098878512+0.1250059821189283j, -0.0365707665795836+0.0115406120327479j]), array([0.1866186119814453-0.273702789100797j , 0.0951662765126204+0.2047672440342183j, 0.326436162342683 -0.1621559972488908j, 0.1033676580865455+0.099812405857126j ]), array([ 0.1641949488505123-0.0089200990587891j,  0.0503781134521   +0.1575344759836166j, -0.2139317236787595+0.1669674902930154j, -0.0712568445097683-0.1669039855498488j]), array([-0.0993420265648789-0.0208796654320543j, -0.0355284278341295-0.1556811426027595j,  0.2562466904396986-0.3147632426010716j,  0.3476852101643126+0.1253128542771293j])]

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

time_dependent_potential=0.1*points #I. e. 0.1*x - very strong field
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

if tmax==0 and method=="DFT":

    orbitals=make_orbitals_numba(lincoeff_initial,gaussian_nonlincoeffs,functions)
    print(calculate_overlapmatrix(orbitals,weights))

    kin_orbitals=make_orbitals_numba(lincoeff_initial,gaussian_nonlincoeffs,minus_half_laplacians)
    for orbital in orbitals:
        plt.plot(points,abs(orbital))
    plt.xlim(-10,10)
    plt.show()
    
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
    plt.plot(points,density)
    plt.xlim(-10,10)
    plt.show()
rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,
                                timestep=dt,points=points,nfrozen=nfrozen,t=tmax,norms=norms_initial,params_previous=gaussian_nonlincoeffs_prev,method=method)

rothepropagator.propagate_nsteps(Tmax,maxiter)
