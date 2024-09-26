import numpy as np
from numpy import pi,sqrt
import matplotlib.pyplot as plt
import os
name_gauss="linear_basis_orbitals_unitary.npz"
name_grid="grid_solution.npz"
orbitals_grid=np.load(name_grid)
grid_grid=orbitals_grid['gridpoints']
Cvals_grid=orbitals_grid['Cvals']
orbitals_gauss=np.load(name_gauss)
grid_gauss=orbitals_gauss['gridpoints']
Cvals_gauss=orbitals_gauss['Cvals']
times=np.linspace(0,310,Cvals_gauss.shape[0])
dg=grid_grid[1]-grid_grid[0]
for i in range(0,Cvals_gauss.shape[0],10):
    plt.figure()
    plt.title('Time: %.3f'%times[i])
    plt.plot(grid_grid,abs(Cvals_grid[i,:,0]/sqrt(dg))**2,label='orbital 1 grid')
    plt.plot(grid_grid,abs(Cvals_grid[i,:,1]/sqrt(dg))**2,label='orbital 1 grid')
    plt.plot(grid_gauss,abs(Cvals_gauss[i,0,:])**2,label='orbital 1 gauss')
    plt.plot(grid_gauss,abs(Cvals_gauss[i,1,:])**2,label='orbital 2 gauss')
    plt.legend()
    try:
        plt.savefig("both/orbitals_t=%.3f.png"%times[i])
    except:
        os.mkdir("both")
        plt.savefig("both/orbitals_t=%.3f.png"%times[i])
    plt.close()