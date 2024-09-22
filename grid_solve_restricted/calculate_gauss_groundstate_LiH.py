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
from opt_einsum import contract as einsum
from scipy import linalg
import time
import chaospy
from scipy.optimize import minimize
from numpy import array,sqrt
from numpy import exp
from numba import cuda

np.set_printoptions(linewidth=300)
def gaussian_quadrature(a, b, n):
    x, w = roots_legendre(n)
    points = 0.5 * (b - a) * x + 0.5 * (b + a)
    weights = 0.5 * (b - a) * w   
    return points, weights


n = 200
a, b = -10, 10
points, weights = gaussian_quadrature(a, b, n)
sqrt_weights=np.sqrt(weights)

R_list=[-1.15, 1.15]
Z_list=[3,1]
alpha=0.5
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

"""
x, a, b, p, q, creal, cimag = sp.symbols('x a b p q creal cimag',real=True)
def sympy_to_python(expr):
    return sp.lambdify((x, a, b, p, q), expr, modules=['numpy'])
def sympy_to_python_xonly(expr):
    return sp.lambdify(x, expr, modules=['numpy'])
"""
#gauss_expr=sp.exp(-(a**2+1j*b)*(x-q)**2+1j*p*(x-q))
#minus_half_laplace_expr=sp.simplify(-0.5*sp.diff(gauss_expr,x,2))

#Initial Parameters from fit_gaussians_to_initial_orbitals.py
"""
params0=[np.float64(1.7313261474958146), np.float64(0.10821645278347743), np.float64(-1.2910941625922085), np.float64(0.20892482699888315), np.float64(0.4981681505938854), np.float64(0.0481965102828324), np.float64(0.6047831402605106), np.float64(-0.12317719386540131), np.float64(0.7310216784245945), np.float64(-1.68942967591808), np.float64(-0.11171375223711236), np.float64(-0.40018603716328593), np.float64(0.9776067505414038), np.float64(0.36088184192239114), np.float64(-1.1485330743923916), np.float64(-1.288518725300041), np.float64(-0.011337982784625984), np.float64(0.14007062103670387), np.float64(1.3259799521738802), np.float64(0.03235138015250231), np.float64(-0.11307079404800245), np.float64(-0.1784697640976569), np.float64(-0.8801885547096829), np.float64(0.7702868337593587), np.float64(1.6212742395117832), np.float64(0.9134314727321242), np.float64(-0.965041609991576), np.float64(-0.16622703353406626), np.float64(0.22562722365141613), np.float64(-0.7359915754998531), np.float64(0.6155838298452647), np.float64(-0.15709120909789054), np.float64(-0.9316561147405571), np.float64(0.16782006597033602), np.float64(-0.2730991793188743), np.float64(-0.1421833099515827), np.float64(0.48880544502010836), np.float64(0.03010690919845806), np.float64(-0.12479446485414844), np.float64(-0.708955887606784), np.float64(1.0668338921462117), np.float64(0.1796929355262306), np.float64(-0.5310253307737399), np.float64(0.18995893818503626), np.float64(0.3049767835902375), np.float64(-0.13457555587939615), np.float64(-0.22645713827542588), np.float64(-0.022577855548405898), np.float64(0.3111163819770864), np.float64(0.1193223653248176), np.float64(0.3458187984555832), np.float64(0.2728926129111036), np.float64(-0.008866439981235173), np.float64(0.00476794224166026), np.float64(-0.318045590309666), np.float64(0.14319089004215707), np.float64(0.17105067148984104), np.float64(0.47990034440714524), np.float64(0.001993210579278009), np.float64(0.006529976810865403)]

params1=[np.float64(1.504347636646048), np.float64(-0.09873685009888035), np.float64(-0.8864151631650046), np.float64(-0.047087665981545145), np.float64(0.5698972710036039), np.float64(-0.023188811749784516), np.float64(0.5277212776205397), np.float64(-0.07897242106875263), np.float64(0.34385524876816626), np.float64(-1.5237066001187636), np.float64(-0.2694594418441665), np.float64(-0.3781879259315648), np.float64(0.7636445276525089), np.float64(0.3614410349005539), np.float64(-0.7891934869641305), np.float64(-1.4686265517738166), np.float64(-0.16903354896292494), np.float64(0.09866933326714471), np.float64(1.0775241897595835), np.float64(0.022281233216022735), np.float64(0.08098682451509065), np.float64(-0.27421421460827355), np.float64(-0.8341123518064296), np.float64(0.6577145968509026), np.float64(1.39539624401962), np.float64(0.6591364971644158), np.float64(-0.9227548558683377), np.float64(-0.5521862640405895), np.float64(0.24152544990590855), np.float64(-0.5511502858464317), np.float64(0.4498337980593697), np.float64(-0.05243736009132614), np.float64(-0.9252785387468863), np.float64(0.10993235955838533), np.float64(0.09212004589722649), np.float64(0.008625456512061678), np.float64(0.41339772413226894), np.float64(-0.024597993510213776), np.float64(-0.07868166505746746), np.float64(-0.23530631332863924), np.float64(0.7615131697151937), np.float64(0.3886260836206984), np.float64(-0.7495678443875794), np.float64(0.1172475764505243), np.float64(0.5207984719504867), np.float64(-0.7594954081774227), np.float64(-0.2926177732629864), np.float64(-0.24958043471548128), np.float64(0.24645650346690456), np.float64(-0.005706379943131093), np.float64(0.4153535840167152), np.float64(-0.14894031133835187), np.float64(0.060759323418712395), np.float64(0.05366013570885475), np.float64(-0.2754869737301915), np.float64(0.0008491768544033885), np.float64(0.32804888580807834), np.float64(0.0940476778490136), np.float64(-0.05020302538156863), np.float64(-0.22500310862866352)]
params0_reshaped=np.array(params0).reshape((len(params0)//6,6))
params1_reshaped=np.array(params1).reshape((len(params1)//6,6))

gaussian_nonlincoeffs=np.concatenate((params0_reshaped[:,:4],params1_reshaped[:,:4]))
"""
gaussian_nonlincoeffs=[array([ 1.73419,  0.11343, -1.27528,  0.20927]), array([ 0.58529, -0.10182,  0.7117 , -1.76771]), array([ 2.11929,  2.17511, -0.88734, -1.16533]), array([ 1.29685,  0.04576, -0.1202 , -0.17472]), array([ 1.62669,  0.90041, -0.96891, -0.1685 ]), array([ 0.6496 , -0.11852, -0.91406,  0.3228 ]), array([ 0.49845,  0.03352, -0.13568, -0.78859]), array([-0.54314,  0.17441,  0.32153, -0.12239]), array([1.90321, 1.51622, 0.24345, 1.22702]), array([1.3594 , 0.2869 , 0.15527, 0.46575]), array([ 1.5183 , -0.10313, -0.91311, -0.04304]), array([ 0.5269 , -0.08562,  0.34713, -1.45937]), array([ 0.75832,  0.35972, -0.7476 , -1.51205]), array([ 1.13422,  0.20182,  0.17226, -0.24265]), array([ 1.4865 ,  0.50411, -1.10446, -0.51211]), array([ 0.4632 , -0.04265, -0.92913,  0.11099]), array([ 0.41095, -0.02081, -0.06822, -0.23986]), array([-0.76367,  0.09731,  0.46446, -0.79924]), array([ 0.24858, -0.00625,  0.41402, -0.16231]), array([-0.27584,  0.00092,  0.32731,  0.08046])]
for k in range(1):
    pass
    #params=[1/sqrt(2)+0.1*np.random.rand()-0.05,0,20*np.random.rand()-10,0]
    #gaussian_nonlincoeffs.append(params)

gaussian_nonlincoeffs=np.array(gaussian_nonlincoeffs)

num_gauss=gaussian_nonlincoeffs.shape[0]
def gauss(x,a,b,p,q):
    bredde=(a**2 + 1j*b)
    qminx=q-x
    return np.exp(-qminx*(1j*p + bredde*qminx))
def minus_half_laplacian(x,a,b,p,q):
    bredde=(a**2 + 1j*b)
    qminx=q-x
    return (bredde - 2.0*(0.5j*p + bredde*qminx)**2)*gauss(x,a,b,p,q)
@jit(nopython=True, fastmath=True)
def exp_pade(x):
    numerator = 1680 + 840*x + 180*x**2 + 20*x**3 + x**4
    denominator = 1680 - 840*x + 180*x**2 - 20*x**3 + x**4
    return numerator / denominator

@jit(nopython=True, fastmath=True,cache=True)
def gauss_and_minushalflaplacian(x, a, b, p, q):
    bredde = a**2 + 1j*b
    qminx = q - x
    jp=1j*p
    gaussval = exp(-qminx * (jp + bredde*qminx))
    minus_half_laplace = (bredde - 0.5 * (jp + 2*bredde*qminx)**2) * gaussval
    return gaussval, minus_half_laplace

@jit(nopython=True, fastmath=True,cache=True)
def setupfunctions(gaussian_nonlincoeffs,points):
    num_gauss = len(gaussian_nonlincoeffs)
    functions = np.empty((num_gauss, len(points)), dtype=np.complex128)
    minus_half_laplacians = np.empty((num_gauss, len(points)), dtype=np.complex128)
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
potential_grid=calculate_potential(Z_list,R_list,alpha,points)


onebody_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)
overlap_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)

wT=weights.T
def calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid,wT):
    for i in range(num_gauss):
        i_conj=np.conj(functions[i])
        for j in range(num_gauss):
            integrand_minus_half_laplace=i_conj*minus_half_laplacians[j]
            integrand_overlap=i_conj*functions[j]
            integrand_potential=integrand_overlap*potential_grid
            overlap_integraded=wT@integrand_overlap
            ij_integrated=wT@integrand_minus_half_laplace+wT@integrand_potential
            onebody_matrix[i,j]=ij_integrated
            overlap_matrix[i,j]=overlap_integraded
    return onebody_matrix,overlap_matrix
e_e_grid=e_e_interaction(points)
@jit(nopython=True,fastmath=True)
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
    s_eigenvalues, s_eigenvectors = linalg.eigh(S)
    X = linalg.inv(linalg.sqrtm(S))

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
def calculate_energy(gaussian_nonlincoeffs,return_all=False,C_init=None,maxiter=100):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    onebody_matrix,overlap_matrix=calculate_onebody_and_overlap(functions,minus_half_laplacians,potential_grid,wT)
    twobody_integrals=calculate_twobody_integrals_numba(np.ascontiguousarray(functions), e_e_grid, weights, num_gauss)
    repulsion_contribution=0
    for i in range(len(Z_list)):
        for j in range(i+1,len(Z_list)):
            repulsion_contribution+=Z_list[i]*Z_list[j]/np.abs(R_list[i]-R_list[j])
    E,C,F,epsilon=restricted_hartree_fock(overlap_matrix,onebody_matrix,twobody_integrals,4,C_init=C_init,max_iterations=maxiter)
    #print(epsilon)
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
def make_orbitals(C,gaussian_nonlincoeffs):
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs.reshape((num_gauss,4)),points)
    return make_orbitals_numba(C,gaussian_nonlincoeffs,functions)
@jit(nopython=True,fastmath=True)
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
    return calculate_Fgauss_fast(np.array(fockOrbitals),gaussian_nonlincoeffs,num_gauss,time_dependent_potential,np.array(functions),np.array(minus_half_laplacians))
weighted_e_e_grid = e_e_grid * weights[:, np.newaxis]

@jit(nopython=True,fastmath=True)
def calculate_Fgauss_fast(fockOrbitals,gaussian_nonlincoeffs,num_gauss,time_dependent_potential,functions,minus_half_laplacians):
    nFock=len(fockOrbitals)
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    Fgauss=np.zeros_like(functions)
    for i in range(num_gauss):
        Fgauss[i] += minus_half_laplacians[i]  # minus_half_laplacian term
        Fgauss[i] += potential_grid * functions[i]  # Potential term
        
        if time_dependent_potential is not None:
            Fgauss[i] += time_dependent_potential * functions[i]
        for j in range(nFock):
            # Coulomb term (equivalent to first einsum)
            coulomb_term = np.dot(np.conj(fockOrbitals[j]) * fockOrbitals[j], weighted_e_e_grid)
            Fgauss[i] += 2 * coulomb_term * functions[i]

            # Exchange term (equivalent to second einsum)
            exchange_term = np.dot(np.conj(fockOrbitals[j]) * functions[i], weighted_e_e_grid)
            Fgauss[i] -= exchange_term * fockOrbitals[j]
    return Fgauss
@jit(nopython=True,fastmath=True)
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

gaussian_nonlincoeffs=gaussian_nonlincoeffs.flatten()

"""

best_orbitals=make_orbitals(C,gaussian_nonlincoeffs)

FockGauss=calculate_Fgauss(best_orbitals,gaussian_nonlincoeffs)
FockGausssq=calculate_Fgauss(FockGauss,gaussian_nonlincoeffs)

Fock_times_Orbitals=calculate_Ftimesorbitals(C,FockGauss)
orn=1   
for orb in [0,1]:
    plt.plot(points,np.abs(best_orbitals[orb]*epsilon[orb]),label="|Basis F*psi(%d)|"%orb)
    plt.plot(points,np.abs(Fock_times_Orbitals[orb]),label="|Exact F*psi(%d)|"%orb)
plt.legend()
print("Energy: ",E)
plt.savefig('Fpsi.png', dpi=500)

plt.show()
sol=minimize(calculate_energy,gaussian_nonlincoeffs,args=(False,C,15),method='Powell',options={'maxiter':20000,"eps":5e-9})
nonlin_params=sol.x
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
def calculate_x_expectation(C,gaussian_nonlincoeffs):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
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
class Rothe_evaluator:
    def __init__(self,old_params,old_lincoeff,time_dependent_potential,timestep):
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
        self.orbitals_that_represent_Fock=make_orbitals(self.old_lincoeff,self.old_params) #Orbitals that define the Fock operator; which are the old orbitals

        self.old_action=self.calculate_Adagger_oldOrbitals() #Essentially, the thing we want to approximate with the new orbitals
    def calculate_Adagger_oldOrbitals(self):
        
        fock_act_on_old_gauss=calculate_Fgauss(self.orbitals_that_represent_Fock,gaussian_nonlincoeffs,num_gauss=self.nbasis,time_dependent_potential=self.pot) #Act with the OLD Fock operator on the OLD Gaussians
        Fock_times_Orbitals=calculate_Ftimesorbitals(self.old_lincoeff,fock_act_on_old_gauss)
        rhs=self.orbitals_that_represent_Fock-1j*self.dt/2*Fock_times_Orbitals
        return rhs
    def calculate_A_newOrbitals(self,new_params,new_lincoeff):
        new_orbitals=make_orbitals(new_lincoeff,new_params)

        #This one below is the most time-consuming. It is the one that can be sped up the most
        fock_act_on_new_gauss=calculate_Fgauss(self.orbitals_that_represent_Fock,new_params,num_gauss=self.nbasis,time_dependent_potential=self.pot) #Act with the OLD Fock operator on the NEW Gaussians
        Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)

        lhs=new_orbitals+1j*self.dt/2*Fock_times_Orbitals

        return lhs
    def calculate_rothe_error(self,full_new_params,n_gauss):
        old_action=self.old_action
        #The first n_gauss*n_orbs parameters are the real parts of full_new_params
        #The next n_gauss*n_orbs parameters are the imaginary parts of full_new_params
        #The rest are the non-linear parameters
        new_params,new_lincoeff=make_Cmat_and_nonlin_from_params(full_new_params,n_gauss,self.norbs)
        new_action=self.calculate_A_newOrbitals(new_params,new_lincoeff)
        orbital_errors=abs(new_action-old_action)
        weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
        rothe_error=np.linalg.norm(weighted_errors)
        return rothe_error
    def calculate_numerical_rothe_jacobian(self,full_new_params,n_gauss,eps=1e-8):
        jac=np.zeros_like(full_new_params)
        old_action=self.old_action
        new_params,new_lincoeff=make_Cmat_and_nonlin_from_params(full_new_params,n_gauss,self.norbs)
        new_orbitals=make_orbitals(new_lincoeff,new_params)
        fock_act_on_new_gauss=fock_act_on_new_gauss_old=calculate_Fgauss(self.orbitals_that_represent_Fock,new_params,num_gauss=self.nbasis,time_dependent_potential=self.pot)
        Fock_times_Orbitals=calculate_Ftimesorbitals(new_lincoeff,fock_act_on_new_gauss)
        new_action=new_orbitals+1j*self.dt/2*Fock_times_Orbitals
        orbital_errors=abs(new_action-old_action)
        weighted_errors=np.einsum("ij,j->ij",orbital_errors,sqrt_weights)
        re=np.linalg.norm(weighted_errors)
        
        for i in range(len(full_new_params)):
            full_new_params_copy=full_new_params.copy()
            full_new_params_copy[i]+=eps
            new_params,new_lincoeff=make_Cmat_and_nonlin_from_params(full_new_params_copy,n_gauss,self.norbs)
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
            rothe_error_pdE=np.linalg.norm(weighted_errors)
            jac[i]=(rothe_error_pdE-re)/eps
        return jac

class Rothe_propagation:
    def __init__(self,params_initial,lincoeffs_initial,pulse,timestep,points):
        self.nbasis=lincoeffs_initial.shape[0]
        self.norbs=lincoeffs_initial.shape[1]
        self.pulse=pulse
        self.dt=timestep
        self.lincoeffs=lincoeffs_initial
        self.params=params_initial

    def propagate(self,t):
        initial_lincoeffs=self.lincoeffs
        initial_params=self.params
        self.time_dependent_potential=self.pulse(t+self.dt/2)*points
        F,S=calculate_Fock_and_overlap(initial_lincoeffs,initial_params,time_dependent_potential=self.time_dependent_potential)
        Spf=S+1j*dt/2*F
        Smf=S-1j*dt/2*F
        C_linear_update=np.linalg.solve(Spf,Smf@initial_lincoeffs) #Initial guess for the linear coefficients in a basisj of the new Gaussians
        C_flat=C_linear_update.flatten()
        initial_full_new_params=np.concatenate((C_flat.real,C_flat.imag,initial_params))
        rothe_evaluator=Rothe_evaluator(initial_params,initial_lincoeffs,self.time_dependent_potential,dt)
        jac_init=rothe_evaluator.calculate_numerical_rothe_jacobian(initial_full_new_params,num_gauss)
        dg=1e-8
        hess_inv0=np.diag(1/abs(jac_init+dg))
        sol=minimize(rothe_evaluator.calculate_rothe_error,initial_full_new_params,jac=rothe_evaluator.calculate_numerical_rothe_jacobian,args=(num_gauss,),method='BFGS',options={'maxiter':30, "hess_inv0": hess_inv0})
        new_rothe_error=rothe_evaluator.calculate_rothe_error(sol.x,num_gauss)
        print("New Rothe error: ",new_rothe_error)
        new_params,new_lincoeff=make_Cmat_and_nonlin_from_params(sol.x,num_gauss,self.norbs)
        self.params=new_params
        self.lincoeffs=new_lincoeff
    def propagate_nsteps(self,nsteps):
        x_expectations=[]
        for i in range(nsteps):
            self.propagate(i*self.dt)
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
dt=0.02
time_dependent_potential=0.1*points #I. e. 0.1*x - very strong field

E,lincoeff_initial,epsilon=calculate_energy(gaussian_nonlincoeffs,return_all=True)

x_expectation_t0=calculate_x_expectation(lincoeff_initial,gaussian_nonlincoeffs)
E0 = 0#0.0534   # Maximum field strength
omega = 0.06075  # Laser frequency
t_c = 2 * np.pi / omega  # Optical cycle
n_cycles = 1

td = n_cycles * t_c  # Duration of the laser pulse
tfinal = td  # Total time of the simulation
print(tfinal)
t=np.linspace(0,tfinal,1000)
fieldfunc=laserfield(E0, omega, td)
plt.plot(t, fieldfunc(t))
plt.show()
rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,timestep=dt,points=points)
nsteps=2500
x_expectations=rothepropagator.propagate_nsteps(nsteps)
x_expectations=[x_expectation_t0]+x_expectations
plt.plot(np.linspace(0,nsteps*dt,nsteps+1),x_expectations)
plt.show()