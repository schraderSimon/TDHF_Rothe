import numpy as np
from numpy import pi,sqrt
import matplotlib.pyplot as plt
import os
import scipy
import sys
import re
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
molecule=sys.argv[1]

fieldstrengh=float(sys.argv[2])
#num_gauss_init=int(sys.argv[3])
#num_gauss=int(sys.argv[3])
try:
    name_grid="/home/simonsch/projects/TDHF/grid-methods/examples/grid_solution_%s_%.3f.npz"%(molecule,fieldstrengh)
    orbitals_grid=np.load(name_grid)
    times_grid=orbitals_grid['times']
    dipmom_grid=orbitals_grid['xvals']
except:
    print("No grid solution found")

times_rothe_list=[]
errors_rothe_list=[]
dipmom_rothe_list=[]
directory = os.getcwd()
pattern = r"Rothe_wavefunctions_[A-Za-z0-9]+_\d+\.\d+_\d+_(\d+)_(\d+)_(\d\.\d+e[+-]\d+)\.npz"
files = [f for f in os.listdir(directory) if f.startswith("Rothe_wavefunctions_%s_%.4f_"%(molecule,fieldstrengh))]
lenghts=[]
maxiters=[]
epsilons=[]
initlengths=[]
for i, file in enumerate(files):
    match = re.match(pattern, file)
    N_init = int(match.group(1))
    maxiter = int(match.group(2))
    epsilon = float(match.group(3))
    initlengths.append(N_init)
    epsilons.append(epsilon)
    maxiters.append(maxiter)
    rothe=np.load(file)
    times_Rothe=rothe["times"]
    dipmom_Rothe=rothe["xvals"]
    errors_Rothe=rothe["rothe_errors"]
    lenghts.append(len(times_Rothe))
    errors_rothe_list.append(errors_Rothe)
    times_rothe_list.append(times_Rothe)
    dipmom_rothe_list.append(dipmom_Rothe)
    


minlen=min(lenghts)
print("Time: %.2f"%times_rothe_list[i][:minlen][-1])
 
for i,rothe_errors in enumerate(errors_rothe_list):
    print("RE, timesteps=%3d,number_gaussians=%d, epsilon=%.3e: %f"%(maxiters[i],initlengths[i],epsilons[i],np.sum(errors_rothe_list[i][:minlen])))

for i,file in enumerate(files):
    # Match the pattern
    match = re.match(pattern, file)

    N_init = int(match.group(1))
    maxiter = int(match.group(2))
    epsilon = float(match.group(3))
    plt.plot(times_rothe_list[i],dipmom_rothe_list[i],label=r"$N_{init}=%d$,$it=%d$,$\varepsilon=%.3e$"%(N_init,maxiter,epsilon))
try:
    plt.plot(times_grid,dipmom_grid,label="Grid")
except:
    print("No grid solution found")
    
plt.legend()
plt.savefig("plots/dipole_moment_%s_%f.pdf"%(molecule,fieldstrengh))

plt.show()
plt.close()



for i,file in enumerate(files):
    match = re.match(pattern, file)

    N_init = int(match.group(1))
    maxiter = int(match.group(2))
    epsilon = float(match.group(3))
    plt.plot(times_rothe_list[i],errors_rothe_list[i],label=r"$N_{init}=%d$,$it=%d$,$\varepsilon=%.3e$,"%(N_init,maxiter,epsilon))

plt.legend()
plt.savefig("plots/errors_%s_%f.pdf"%(molecule,fieldstrengh))
plt.show()
