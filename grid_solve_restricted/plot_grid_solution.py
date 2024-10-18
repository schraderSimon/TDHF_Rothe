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
    omega = (
        scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(time_points)))
        * 2
        * np.pi
        / dt
    )

    return omega, Px
fieldstrengh=float(sys.argv[1])
num_gauss=int(sys.argv[3])


name_grid="/home/simonsch/projects/TDHF/grid-methods/examples/grid_solution_LiH_%.3f.npz"%fieldstrengh
orbitals_grid=np.load(name_grid)
times_grid=orbitals_grid['times']
dipmom_grid=orbitals_grid['xvals']
plt.plot(times_grid,dipmom_grid,label="Grid")


name_gauss="Rothe_wavefunctions%.4f_%d_%d_0_LiH.npz"%(fieldstrengh,int(sys.argv[2]),int(sys.argv[3]))
gauss=np.load(name_gauss)
times_gauss=gauss["times"]
dipmom_gauss=gauss["xvals"]
errors_gauss=gauss["rothe_errors"]
plt.plot(times_gauss,dipmom_gauss,label="Linear Rothe")

name_Rothe="Rothe_wavefunctions%.4f_%d_%d_%d_LiH.npz"%(fieldstrengh,int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
rothe=np.load(name_Rothe)
times_Rothe=rothe["times"]
dipmom_Rothe=rothe["xvals"]
errors_Rothe=rothe["rothe_errors"]
plt.plot(times_Rothe,dipmom_Rothe,label="Full Rothe")

print("Cumulative Rothe error (linear):",np.sum(errors_gauss))
print("Cumulative Rothe error (full):",np.sum(errors_Rothe))

plt.legend()
plt.savefig("dipole_moment_%f_%d.pdf"%(fieldstrengh,num_gauss))

plt.show()
plt.close()

plt.plot(times_gauss,errors_gauss,label="Linear Rothe")

plt.plot(times_Rothe,errors_Rothe,label="Full Rothe")

plt.legend()
plt.savefig("errors_%f_%d.pdf"%(fieldstrengh,num_gauss))
plt.show()
try:
    ltr=times_Rothe.shape[0]
except:
    sys.exit()
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
plt.savefig("spectra%f_%d.pdf"%(fieldstrengh,num_gauss))
plt.show()
plt.close()
