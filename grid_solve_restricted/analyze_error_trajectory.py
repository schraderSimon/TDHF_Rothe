import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
tmin=float(sys.argv[1])
tmax=float(sys.argv[2])
times=np.linspace(tmin,tmax,int(20*(tmax-tmin)+1+1e-4))
print(times)
error_trajectories=[]
error_trajectories_t0=[]
for time in times:
    infile="error_trajectory/trajectory_%.2f.npz"%time
    try:
        error_trajectory=np.load(infile)
        error_trajectory=error_trajectory['f_storage']
        error_trajectories.append(error_trajectory**2)
        error_trajectories_t0.append(error_trajectory[0]**2)
    except:
        pass
smallest_error=np.max(error_trajectories_t0)
rescaled_trajectories=[]
for error_trajectory in error_trajectories:
    #Make all start at the same value
    rescaled_error_trajectory=error_trajectory*(smallest_error/error_trajectory[0])
    rescaled_trajectories.append(rescaled_error_trajectory)

ret0=rescaled_trajectories[0]
nmax=len(ret0)-1#min(100,len(ret0)-3)
mean_rets=[]
ret_avgval_initial=0
ret_avgval_final=0
for ret in rescaled_trajectories:
    ret_avgval_initial+=ret[0]
    ret_avgval_final+=ret[-1]
ret_avgval_initial/=len(rescaled_trajectories)
ret_avgval_final/=len(rescaled_trajectories)
for ret in rescaled_trajectories:
    print(nmax,len(ret),len(ret0))
    if len(ret)<nmax:
        ret=np.pad(ret,(0,nmax+2-len(ret)),'constant',constant_values=(ret[-1],ret[-1]))
    a=(ret0[nmax-1]-ret0[0])/(ret[nmax-1]-ret[0])
    b=ret0[0]-a*ret[0]
    mean_rets.append(sqrt(ret[:nmax]*a+b))
for mret in mean_rets:
    plt.plot(mret)
median_rets = np.median(mean_rets, axis=0)
percentile_25 = np.percentile(mean_rets, 25, axis=0)
percentile_75 = np.percentile(mean_rets, 75, axis=0)

plt.plot(median_rets, label='Median')
plt.fill_between(np.arange(nmax), percentile_25, percentile_75, alpha=0.5, label='25th-75th Percentile')
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()