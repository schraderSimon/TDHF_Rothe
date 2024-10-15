import numpy as np
from numpy import pi,sqrt
import matplotlib.pyplot as plt
import os
import scipy
import sys
def compute_hhg_spectrum(time_points, dipole_moment, hann_window=False):

    dip = scipy.signal.detrend(dipole_moment, type="constant")
    if hann_window:
        Px = (
            np.abs(
                scipy.fftpack.fftshift(
                    scipy.fftpack.fft(
                        dip * np.sin(np.pi * time_points / time_points[-1]) ** 2
                    )
                )
            )
            ** 2
        )
    else:
        Px = np.abs(scipy.fftpack.fftshift(scipy.fftpack.fft(dip))) ** 2

    dt = time_points[1] - time_points[0]
    print(dt)
    omega = (
        scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(time_points)))
        * 2
        * np.pi
        / dt
    )

    return omega, Px
fieldstrengh=float(sys.argv[1])
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
    name_Rothe="Rothe_wavefunctions%.3f_%d_%d.npz"%(fieldstrengh,int(sys.argv[2]),int(sys.argv[3]))

    rothe=np.load(name_Rothe)

    times_Rothe=rothe["times"]
    dipmom_Rothe=rothe["xvals"]

    plt.plot(times_Rothe,dipmom_Rothe,label="Rothe_linear")
except:
    raise ValueError("No Rothe data found")
ltr=times_Rothe.shape[0]
plt.legend()
plt.savefig("dipole_moment.png")
plt.show()
plt.close()
grid_omega,grid_Px=compute_hhg_spectrum(times_grid[:ltr],dipmom_grid[:ltr])
gauss_omega,gauss_Px=compute_hhg_spectrum(times_gauss[:ltr],dipmom_gauss[:ltr])
Rothe_omega,Rothe_Px=compute_hhg_spectrum(times_Rothe,dipmom_Rothe)
plt.plot(grid_omega/0.05,grid_Px,label="Grid")
plt.plot(gauss_omega/0.05,gauss_Px,label="Gauss")
plt.plot(Rothe_omega/0.05,Rothe_Px,label="Rothe")
plt.xlim(0,10/0.05)
plt.ylim(1e-8,10*max(grid_Px))
plt.yscale("log")
plt.legend()
plt.savefig("spectra.png")
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