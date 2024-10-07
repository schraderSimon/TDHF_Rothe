import numpy as np
from numpy import pi,sqrt
import matplotlib.pyplot as plt
import os
import sys
fieldstrengh=float(sys.argv[2])
num_gauss=20
name_gauss="linearbasis_%.3f_%d.npz"%(fieldstrengh,num_gauss)
name_grid="grid_solution_%.3f.npz"%fieldstrengh
orbitals_grid=np.load(name_grid)
grid_grid=orbitals_grid['gridpoints']
Cvals_grid=orbitals_grid['Cvals']
dipmom_grid=orbitals_grid['dipole']
orbitals_gauss=np.load(name_gauss)
dipmom_gauss=orbitals_gauss['x']
times_gauss=orbitals_gauss['t']

times_grid=np.linspace(0,310,Cvals_grid.shape[0])

dg=grid_grid[1]-grid_grid[0]
plt.plot(times_grid,dipmom_grid,label="Grid")
plt.plot(times_gauss,dipmom_gauss,label="Gauss")
try:
    name_Rothe="Rothe_wavefunctions%.3f_%d.npz"%(fieldstrengh,int(sys.argv[1]))

    rothe=np.load(name_Rothe)

    times_Rothe=rothe["times"]
    dipmom_Rothe=rothe["xvals"]

    plt.plot(times_Rothe,dipmom_Rothe,label="Rothe_linear")
except:
    pass

plt.legend()
plt.savefig("dipole_moment.png")
plt.show()
plt.close()

"""
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
"""