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
num_gauss_init=int(sys.argv[2])
num_gauss=int(sys.argv[3])
sys_arg_len=len(sys.argv)
maxiters=np.array(sys.argv[4:-1],dtype=int)
print(maxiters)
molecule=sys.argv[-1]
name_grid="/home/simonsch/projects/TDHF/grid-methods/examples/grid_solution_%s_%.3f.npz"%(molecule,fieldstrengh)
orbitals_grid=np.load(name_grid)
times_grid=orbitals_grid['times']
dipmom_grid=orbitals_grid['xvals']
plt.plot(times_grid,dipmom_grid,label="Grid")


name_gauss="Rothe_wavefunctions%.4f_%d_%d_0_%s.npz"%(fieldstrengh,int(sys.argv[2]),int(sys.argv[3]),molecule)
gauss=np.load(name_gauss)
times_gauss=gauss["times"]
dipmom_gauss=gauss["xvals"]
errors_gauss=gauss["rothe_errors"]
plt.plot(times_gauss,dipmom_gauss,label="Linear Rothe")
times_rothe_list=[]
errors_rothe_list=[]
dipmom_rothe_list=[]
len_Rothe=len(errors_gauss)
for maxiter in maxiters:
    name_Rothe="Rothe_wavefunctions%.4f_%d_%d_%d_%s.npz"%(fieldstrengh,int(sys.argv[2]),int(sys.argv[3]),maxiter,molecule)
    rothe=np.load(name_Rothe)
    times_Rothe=rothe["times"]
    dipmom_Rothe=rothe["xvals"]
    errors_Rothe=rothe["rothe_errors"]
    errors_rothe_list.append(errors_Rothe)
    times_rothe_list.append(times_Rothe)
    dipmom_rothe_list.append(dipmom_Rothe)
    plt.plot(times_Rothe,dipmom_Rothe,label="Full Rothe %d"%maxiter)
    len_Rothe=min(len(errors_Rothe),len_Rothe)
for i,maxiter in enumerate(maxiters):
    print("Cumulative Rothe error %d:"%maxiter,np.sum(errors_rothe_list[i][:len_Rothe]))

print("Cumulative Rothe error (linear):",np.sum(errors_gauss[:len_Rothe]))

plt.legend()
plt.savefig("dipole_moment_%f_%d.pdf"%(fieldstrengh,num_gauss))

plt.show()
plt.close()

plt.plot(times_gauss,errors_gauss,label="Linear Rothe")
for i,maxiter in enumerate(maxiters):
    times_Rothe=times_rothe_list[i]
    errors_Rothe=errors_rothe_list[i]
    plt.plot(times_Rothe,errors_Rothe,label="Full Rothe %d"%maxiters[i])

plt.legend()
plt.savefig("errors_%f_%d.pdf"%(fieldstrengh,num_gauss))
plt.show()
try:
    ltr=times_Rothe.shape[0]
except:
    sys.exit()
grid_omega,grid_Px=compute_hhg_spectrum(times_grid[:ltr],dipmom_grid[:ltr])
gauss_omega,gauss_Px=compute_hhg_spectrum(times_gauss[:ltr],dipmom_gauss[:ltr])
for i,maxiter in enumerate(maxiters):
    Rothe_omega,Rothe_Px=compute_hhg_spectrum(times_rothe_list[i],dipmom_rothe_list[i])
    plt.plot(Rothe_omega/0.05,Rothe_Px,label="Rothe %d"%maxiters[i])
plt.plot(grid_omega/0.05,grid_Px,label="Grid")
plt.plot(gauss_omega/0.05,gauss_Px,label="Gauss")
plt.xlim(0,10/0.05)
plt.ylim(1e-8,10*max(grid_Px))
plt.yscale("log")
plt.legend()
plt.savefig("spectra%f_%d.pdf"%(fieldstrengh,num_gauss))
plt.show()
plt.close()
