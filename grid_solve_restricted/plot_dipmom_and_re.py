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
pattern = r"WF_HF_[A-Za-z0-9]+_\d+\.\d+_\d+_(\d+)_(\d+)_(\d\.\d+e[+-]\d+)\.npz"
files = [f for f in os.listdir(directory) if f.startswith("WF_HF_%s_%.4f_"%(molecule,fieldstrengh))]
lenghts=[]
maxiters=[]
epsilons=[]
initlengths=[]
finaltime_gaussians=[]
print(files)
for i, file in enumerate(files):
    match = re.match(pattern, file)
    N_init = int(match.group(1))
    maxiter = int(match.group(2))
    epsilon = float(match.group(3))
    initlengths.append(N_init)
    epsilons.append(epsilon)
    maxiters.append(maxiter)
    rothe=np.load(file)
    params=rothe["params"]
    if molecule=="LiH":
        num_Gaussians_finaltime=len(params[-1])//8
    elif molecule=="LiH2":
        num_Gaussians_finaltime=len(params[-1])//12
    finaltime_gaussians.append(num_Gaussians_finaltime)
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
    print("RE, timesteps=%3d,number_gaussians=%d/%d, epsilon=%.3e: %f"%(maxiters[i],initlengths[i],finaltime_gaussians[i],epsilons[i],np.sum(errors_rothe_list[i][:minlen])))

for i,file in enumerate(files):
    
    # Match the pattern
    match = re.match(pattern, file)

    N_init = int(match.group(1))
    maxiter = int(match.group(2))
    epsilon = float(match.group(3))
    plt.plot(times_rothe_list[i],dipmom_rothe_list[i],label=r"$N_{max}=%d$,$it=%d$,$\varepsilon=%.3e$"%(finaltime_gaussians[i],maxiter,epsilon))
try:
    plt.plot(times_grid,dipmom_grid,label="Grid")
except:
    print("No grid solution found")
    
plt.legend()
plt.savefig("plots/dipole_moment_%s_%f.pdf"%(molecule,fieldstrengh))

plt.show()


for i,file in enumerate(files):
    
    # Match the pattern
    match = re.match(pattern, file)

    N_init = int(match.group(1))
    maxiter = int(match.group(2))
    epsilon = float(match.group(3))
    plt.plot(times_rothe_list[i][1:],dipmom_rothe_list[i][1:]-dipmom_rothe_list[i][:-1],label=r"$N_{max}=%d$,$it=%d$,$\varepsilon=%.3e$"%(finaltime_gaussians[i],maxiter,epsilon))
try:
    plt.plot(times_grid[1:],dipmom_grid[1:]-dipmom_grid[:-1],label="Grid")
except:
    print("No grid solution found")
    
plt.legend()
plt.show()


for i,file in enumerate(files):
    match = re.match(pattern, file)

    N_init = int(match.group(1))
    maxiter = int(match.group(2))
    epsilon = float(match.group(3))
    plt.plot(times_rothe_list[i],errors_rothe_list[i],label=r"$N_{max}=%d$,$it=%d$,$\varepsilon=%.3e$,"%(finaltime_gaussians[i],maxiter,epsilon))

plt.legend()
plt.savefig("plots/errors_%s_%f.pdf"%(molecule,fieldstrengh))
plt.show()
omega=0.06075
grid_omega,grid_Px=compute_hhg_spectrum(times_grid[:minlen],dipmom_grid[:minlen])
plt.figure(figsize=(6,2.3)) # make the figure larger
#plt.plot(grid_omega/omega,grid_Px,label="Grid")

if molecule=="LiH" and fieldstrengh-0.0534<1e-5:
    moleculename="LiH"
    labels=[0,"%d Gaussians (Rothe)"%finaltime_gaussians[1],"%d fixed Gaussians"%finaltime_gaussians[2]]
elif molecule=="LiH2" and fieldstrengh-0.0534<1e-5:
    labels=[0,"%d fixed Gaussians"%finaltime_gaussians[1],0,"%d  Gaussians (Rothe)"%finaltime_gaussians[3]]
    moleculename="(LiH)$_2$"
else:
    moleculename=molecule
plotties_x=[grid_omega/omega]
plotties_y=[grid_Px]
plotties_label=["Grid"]
for k,file in enumerate(files):
    i=k
    if molecule=="LiH" and fieldstrengh-0.0534<1e-5:
        if k==0 and molecule=="LiH":
            continue
        elif k in [0,2] and molecule=="LiH2":
            continue
        elif k==1 and molecule=="LiH2":
            i=3
        elif k==3 and molecule=="LiH2":
            i=1
        plotties_label.append(labels[i])
    match = re.match(pattern, file)

    N_init = int(match.group(1))
    maxiter = int(match.group(2))
    epsilon = float(match.group(3))
    Rothe_omega,Rothe_Px=compute_hhg_spectrum(times_rothe_list[i][:minlen],dipmom_rothe_list[i][:minlen])
    #plt.plot(Rothe_omega/omega,Rothe_Px,label=labels[i])
    plotties_x.append(Rothe_omega/omega)
    plotties_y.append(Rothe_Px)
   
    #plt.plot(Rothe_omega/omega,Rothe_Px,label=r"$N_{max}=%d$,$it=%d$,$\varepsilon=%.3e$"%(finaltime_gaussians[i],maxiter,epsilon))
order=[0,1,2][:len(plotties_x)]
styles=["-","--","--"]
colors=["black","red","blue"]
for k in order:
    if len(plotties_label)>1:
        plt.plot(plotties_x[k],plotties_y[k],label=plotties_label[k],linestyle=styles[k],color=colors[k])
    else:
        if k==0:
            plt.plot(plotties_x[k],plotties_y[k],label="Grid",linestyle=styles[k],color=colors[k])
        else:
            plt.plot(plotties_x[k],plotties_y[k])
plt.title(r"HHG Spectrum of 1D-%s, $E_0=%.3f$ a.u."%(moleculename,fieldstrengh))
if fieldstrengh-0.0534<1e-5:
    plt.xlim(1,40)
else:
    plt.xlim(1,150)
if molecule=="LiH":
    plt.ylim(1e-10*max(grid_Px),3*max(grid_Px))
elif molecule=="LiH2":
    plt.ylim(10**(-7.5)*max(grid_Px),3*max(grid_Px))
plt.yscale("log")
plt.legend()
plt.ylabel("Intensity")
plt.xlabel("Harmonic order")
plt.tight_layout()
plt.savefig("plots/hhg_spectrum_%s_%f.pdf"%(molecule,fieldstrengh))
plt.show()