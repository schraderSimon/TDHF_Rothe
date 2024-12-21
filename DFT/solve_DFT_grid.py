import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags
from exchange_correlation_functionals import *
from numba import jit
@jit(nopython=True)
def kinetic_energy(orbitals, dx,T):
    """
    Compute the kinetic energy from the orbitals.
    """
    E_kin = 0
    for orbital in orbitals:
        orbital = np.ascontiguousarray(orbital)
        orbital_gradient = np.dot(T, orbital)
        E_kin += np.sum(orbital * orbital_gradient) * dx
    return E_kin*2 #Factor of 2 accounts for double occupancy
@jit(nopython=True)
def v_ee_coulomb(grids):
    vp = 1 / np.sqrt(grids**2 + 1)
    return vp
@jit(nopython=True)
def hartree_potential(grid,rho,vee=v_ee_coulomb):
    v_h = np.zeros_like(rho)
    for i, xi in enumerate(grid):
        v_h[i] = np.sum(rho*vee(xi-grid) * dx)
    return v_h

def external_V_softCoulomb(grids,Z,pos,a=1):
    def pot(grids):
        vp = -Z[0] /np.sqrt((grids-pos[0])**2+a)
        for i in range(1,len(Z)):
            vp+=-Z[i] /np.sqrt((grids-pos[i])**2+a)
        return vp
    return pot
@jit(nopython=True)
def hartree_energy_function(grids,density):
    return 0.5*np.sum(hartree_potential(grids,density)*density)*dx
def external_potential_energy_function(grids,density,v_ext):
    dx=grids[1]-grids[0]
    return np.sum(density*v_ext(grids))*dx
def make_kinetic_energy_operator(grids):
    dx=grids[1]-grids[0]
    T = (-0.5 / (12 * dx**2)) * diags([-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2], shape=(len(grids), len(grids))).toarray()    
    return np.ascontiguousarray(T)
def total_energy_function(grids,orbitals,density,pos,charge,v_ext,printing=False,nucnuc=True):
        kin=kinetic_energy_function(orbitals)
        ext=external_potential_energy_function(grids,density,v_ext)
        hartree=hartree_energy_function(grids,density)
        exchange_correlation=sum(exc(density)*density)*dx
        if len(charge)>=1:
            nuclear_nuclear_repulsion=sum([charge[i]*charge[j]/np.sqrt((pos[i]-pos[j])**2+1) for i in range(len(charge)) for j in range(i+1,len(charge))])
        else:
            nuclear_nuclear_repulsion=0
        if printing:
            print("Kinetic energy: ",kin)
            print("External potential energy: ",ext)
            print("Hartree energy: ",hartree)
            print("Exchange Corrleation energy: ",exchange_correlation)
            print("Nuclear-nuclear repulsion: ",nuclear_nuclear_repulsion)
        E=kin+ext+hartree+exchange_correlation
        if nucnuc:
            E+=nuclear_nuclear_repulsion
        return E
if __name__ == '__main__':
    energies=[]
    density=None
    #Test 1: H2 molecule
    run_H2=False
    charge=[1,1]
    n_elec=sum(charge)
    n_orbs=n_elec//2
    grids = np.linspace(-30,30,401)
    dx=grids[1]-grids[0]
    T=make_kinetic_energy_operator(grids)
    def kinetic_energy_function(orbitals):
        return kinetic_energy(orbitals,dx,T)
    
    v_ext_func=external_V_softCoulomb
    exc=epsilon_xc
    exchange_correlation_potential=v_xc
    
    xvals=np.linspace(0,5,101)
    if run_H2:
        for x in xvals:
            pos=[0,x]
            v_ext=v_ext_func(grids,charge,pos,a=1)
            onebody_operator=T+np.diag(v_ext(grids))
            eigvals,eigvecs=eigh(onebody_operator)
            orbitals=eigvecs.T/np.sqrt(dx)

            initial_orbitals=orbitals[:n_orbs]
            initial_density=np.sum(initial_orbitals**2,axis=0)*2
            
            """SCF calculation of density and energy"""
            E_init=total_energy_function(grids,initial_orbitals,initial_density,pos,charge,v_ext)

            if density is None:
                pass
                density=initial_density
            density=np.array(density)
            imax=20
            best_energies=[E_init]
            for i in range(imax):
                old_density=density
                effective_potential=v_ext(grids)+hartree_potential(grids,density)+exchange_correlation_potential(density)
                onebody_operator=T+np.diag(effective_potential)
                eigvals,eigvecs=eigh(onebody_operator)
                orbitals=eigvecs.T/np.sqrt(dx)
                orbitals=orbitals[:n_orbs]

                density=np.sum(orbitals**2,axis=0)*2
                energy=total_energy_function(grids,orbitals,density,pos,charge,v_ext)
                if i<imax-1:
                    pass
                    density=(0.1*density+0.9*old_density)
                    #normalize density
                best_energies.append(energy)
            E=total_energy_function(grids,orbitals,density,pos,charge,v_ext,nucnuc=True)
            print(E)
            energies.append(min(best_energies))
        plt.plot(xvals,energies)
        plt.show()
    names=["Be","Be2+","Li+","He"]
    charges=[4,4,3,2]
    n_elecs=[4,2,2,2]
    for k in range(len(names)):
        charge=[charges[k]]
        n_elec=n_elecs[k]
        n_orbs=n_elec//2
        pos=[0]
        v_ext=v_ext_func(grids,charge,pos,a=1)
        onebody_operator=T+np.diag(v_ext(grids))
        eigvals,eigvecs=eigh(onebody_operator)
        orbitals=eigvecs.T/np.sqrt(dx)
        initial_orbitals=orbitals[:n_orbs]
        initial_density=np.sum(initial_orbitals**2,axis=0)*2
        E_init=total_energy_function(grids,initial_orbitals,initial_density,pos,charge,v_ext)

        if density is None:
            pass
            density=initial_density
        density=np.array(density)
        imax=40
        for i in range(imax):
            old_density=density
            effective_potential=v_ext(grids)+hartree_potential(grids,density)+exchange_correlation_potential(density)
            onebody_operator=T+np.diag(effective_potential)
            eigvals,eigvecs=eigh(onebody_operator)
            orbitals=eigvecs.T/np.sqrt(dx)
            orbitals=orbitals[:n_orbs]

            density=np.sum(orbitals**2,axis=0)*2
            energy=total_energy_function(grids,orbitals,density,pos,charge,v_ext)
            if i<imax-1:
                pass
                density=(0.1*density+0.9*old_density)
        E=total_energy_function(grids,orbitals,density,pos,charge,v_ext,nucnuc=True)
        print("Energy of ",names[k],": ",E)