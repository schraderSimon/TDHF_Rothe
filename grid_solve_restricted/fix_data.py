import numpy as np
import scipy
import sys
import re
import matplotlib.pyplot as plt
import os
def find_duplicates(arr, epsilon=1e-4,start=0):
    """
    Finds pairs of duplicate values in 'arr' (floats) where absolute
    difference < epsilon. Assumes at most two copies of any value.
    
    Returns a list of [representative_value, idx1, idx2],
    where 'representative_value' is a rounded display of the duplicate.
    """
    n = len(arr)
    duplicates = []
    used = set()  # to mark indices we've already paired

    for i in range(n):
        if i in used:
            continue
        for j in range(i + 1, n):
            if j in used:
                continue
            
            # Check if arr[i] and arr[j] differ by less than epsilon
            if abs(arr[i] - arr[j]) < epsilon:
                # You can choose how you want to represent 'duplicate_value' in the output:
                #   - the raw value arr[i]
                #   - or a nicely rounded version, e.g. round(arr[i], 5)
                #   - or an integer if you *know* it's effectively an integer
                duplicate_value = round(arr[i], 5)  # adjust precision to taste

                duplicates.append([duplicate_value, i+start, j+start])
                used.add(i)
                used.add(j)
                # We break because we assume at most two copies for any value
                break
    
    return duplicates


infile=np.load("copy_WF_HF_LiH_0.1068_20_24_300_6.000e-01.npz", "r")

for key in infile.keys():
    print(key)
    print(infile[key].shape)
times=infile["times"]
dipole_moment=infile["xvals"]
rothe_errors=infile["rothe_errors"]
params=infile["params"]

deleted_times=[]
deleted_dipole_moment=[]
repeated_time_indicies=[]
skippy=0
indices_to_keep=[]

for i,time in enumerate(times):
    if time<=288:
        deleted_times.append(time)
        deleted_dipole_moment.append(dipole_moment[i])
        indices_to_keep.append(i)
    if time>288:
        index=i+skippy
        try:
            if times[index+1]-times[index] < 0.04999 or times[index+1]-times[index] > 0.05001:
                skippy+=1
                #print(" %f, Time step not equal to 0.0005"%time)
                print(times[index])
            else:
                print(times[index])
            indices_to_keep.append(index)
        except:
            break
#plt.plot(times[indices_to_keep], dipole_moment[indices_to_keep])
#plt.show()
duplicates=find_duplicates(times[20*290:],start=20*290)
begin_index=int(20*290.2)
print(times[20*290])
dipmom_0=dipole_moment[begin_index]
dipole_moment_closer=[]
index_closer=[i for i in range(begin_index+1)]
for duplicate in duplicates:
    dipmom_1=dipole_moment[duplicate[1]]
    dipmom_2=dipole_moment[duplicate[2]]
    if abs(dipmom_0-dipmom_1)<abs(dipmom_0-dipmom_2):
        dipole_moment_closer.append(dipmom_1)
        dipmom_0=dipmom_1
        index_closer.append(duplicate[1])
    else:
        dipole_moment_closer.append(dipmom_2)
        dipmom_0=dipmom_2
        index_closer.append(duplicate[2])
plt.plot(list(times[index_closer]),list(dipole_moment[index_closer]))
for time in times[index_closer]:
    print(time)
plt.show()

out_times=times[index_closer]
out_RE=rothe_errors[index_closer]
out_dipole_moment=dipole_moment[index_closer]
out_params=params[index_closer,:]
out_matrixnorms=infile["norms"][index_closer,:]
out_nbas=infile["nbasis"][index_closer]
np.savez("WF_HF_LiH_0.1068_20_24_300_6.000e-01.npz", times=out_times, xvals=out_dipole_moment, rothe_errors=out_RE, params=out_params, norms=out_matrixnorms, nbasis=out_nbas)