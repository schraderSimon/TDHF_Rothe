import numpy as np
from numpy import pi,sqrt
import matplotlib.pyplot as plt
import os
import sys
num_gauss=int(sys.argv[1])
name_gauss="linear_basis_orbitals_unitary_%d.npz"%num_gauss
name_grid="grid_solution.npz"
name_Rothe="Rothe_linearbasis.npz"
orbitals_grid=np.load(name_grid)
rothe=np.load(name_Rothe)
grid_grid=orbitals_grid['gridpoints']
Cvals_grid=orbitals_grid['Cvals']
dipmom_grid=orbitals_grid['dipole']
orbitals_gauss=np.load(name_gauss)
grid_gauss=orbitals_gauss['gridpoints']
Cvals_gauss=orbitals_gauss['Cvals']
dipmom_gauss=orbitals_gauss['dipole']

times=times_gauss=np.linspace(0,310,Cvals_gauss.shape[0])
times_grid=np.linspace(0,310,Cvals_grid.shape[0])
times_Rothe=rothe["t"]
dipmom_Rothe=rothe["x"]
dg=grid_grid[1]-grid_grid[0]
plt.plot(times_grid,dipmom_grid,label="Grid")
plt.plot(times_gauss,dipmom_gauss,label="Gauss")

plt.plot(times_Rothe,dipmom_Rothe,label="Rothe_linear")
plt.legend()
plt.savefig("dipole_moment.png")
plt.show()
plt.close()

for i in range(0,Cvals_gauss.shape[0],20):
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