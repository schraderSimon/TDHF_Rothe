import numpy as np
from quantum_systems import ODQD, GeneralOrbitalSystem
import scipy
from helper_functions import *
class GHFSolverSystem(object):
    def __init__(self,number_electrons=2,number_basisfunctions=10,
                grid_length=10,num_grid_points=1001,omega=0.25,a=0.25):
        """Initialize the system"""
        self.M=2*number_basisfunctions #Hartree Fock basis is twice the number of basis functions
        self.grid_length=grid_length
        self.num_grid_points=num_grid_points
        self.steplength=(grid_length*2)/(num_grid_points-1) #Step length for Numerical integration or similar
        self.omega=omega
        self.a=a
        self.number_electrons=number_electrons
        self.odho = ODQD(number_basisfunctions, grid_length, num_grid_points, a=a, alpha=1, potential=ODQD.HOPotential(omega))
        anti_symmetrize=True
        self.system=GeneralOrbitalSystem(n=number_electrons,basis_set=self.odho,
                                        anti_symmetrize=anti_symmetrize)
    def setC(self,C=None):
        """Initiate self.C-matrix and self.epsilon"""
        if C is None: #If C is none
            self.C=np.array(np.random.rand(self.M,self.M),dtype=complex128) #Random matrix (not a legal coefficient matrix)
        else:
            self.C=C
        P=self.construct_Density_matrix(self.C)
        F=self.construct_Fock_matrix(P)
        self.epsilon, throwaway = scipy.linalg.eigh(F)

    def construct_Density_matrix(self,C):
        """Construct and return density matrix from self.C"""
        slicy=slice(0,self.number_electrons)
        return np.einsum("ma,va->mv",C[:,slicy],C.conjugate()[:,slicy])

    def construct_Fock_matrix(self,P):
        """Construct Fock matrix"""
        u=np.einsum("ts,msnt->mn",P,self.system.u)
        return self.system.h+u

    def solve(self,tol=1e-8,maxiter=100):
        """Solve the SCF equations

        Parameters
        ----------
        tol: double
            The "convergence tolerance", system has converged if
            np.max(np.abs(P-P_old))<tol
        maxiter: int
            The maximum number of SCF iterations

        Returns
        -----
        bool
            if the system has converged or not after the proecedure has finished
        """
        converged=False #If difference is less than "tol"
        P=self.construct_Density_matrix(self.C)
        for i in range(maxiter):
            F=self.construct_Fock_matrix(P) #Before the if-test to assure it matches the recent Density matrix
            if(i>0):
                convergence_difference=np.max(np.abs(P-P_old))
                if (convergence_difference<tol):
                    converged=True
                    break
            P_old=P
            self.epsilon, self.C = scipy.linalg.eigh(F)
            P=self.construct_Density_matrix(self.C)
        return converged

    def get_energy(self):
        """Returns the ground-state energy <H>"""
        energy=0
        P=self.construct_Density_matrix(self.C)
        tot_vec=self.system.h+self.construct_Fock_matrix(P)
        energy=0.5*np.einsum("nm,mn->",P,tot_vec)
        return energy

class RHFSolverSystem(GHFSolverSystem):
    """A restricted HF solver

    This class is NOT capable of time development, and many functions from the
    GHFSolverSystem do not work. Only the SCF-algorithm works,
    that is, constructing the SCF-C-matrix, and energy-construction.
    A call to the function "get_full_C", which then needs to be given as C-matrix
    to a GHFSolverSystem, is required for further analysis and applications.
    """
    def __init__(self,number_electrons=2,number_basisfunctions=10,
                grid_length=10,num_grid_points=1001,omega=0.25,a=0.25):
        self.M=number_basisfunctions #Hartree Fock basis is twice the number of basis functions
        self.grid_length=grid_length
        self.num_grid_points=num_grid_points
        self.steplength=(grid_length*2)/(num_grid_points-1) #Step length for Numerical integration or similar
        self.omega=omega
        self.a=a
        self.number_electrons=number_electrons
        self.system=ODQD(number_basisfunctions, grid_length, num_grid_points, a=a, alpha=1, potential=ODQD.HOPotential(omega))

    def construct_Density_matrix(self,C):

        slicy=slice(0,int(self.number_electrons/2)) #Slightly different expression
        return 2*np.einsum("ma,va->mv",C[:,slicy],C.conjugate()[:,slicy]) #Eq. 3.145

    def construct_Fock_matrix(self,P):
        """Construct Fock matrix, explicitely considering direct and exchange term"""
        udirect=np.einsum("ts,msnt->mn",P,self.system.u)
        uexchange=np.einsum("ts,mstn->mn",P,self.system.u)
        return self.system.h+udirect-0.5*uexchange #Also from Szabo-Ostlund

    def get_full_C(self):
        """Return the full coefficient matrix in the spin-orbital basis"""
        C=np.zeros((2*self.M,2*self.M),dtype=np.complex128)
        for i in range(self.M):
            for j in range(self.M):
                C[2*i,2*j]=self.C[i,j] #Same as RHS solution
                C[2*i+1,2*j+1]=self.C[i,j]
        return C