import numpy as np
molecule_names=["LiH","LiH2"]
fieldstrengths=[1.0,4.0]
methods=["DFT"]
maxiter=300
import sys
import math
def make_jobscript(jobscript_name,run_command):
    jobscript=f"""#!/bin/bash
# Job name:
#SBATCH --job-name={jobscript_name}
#
# Project:
#SBATCH --account=nn4654k
#
# Wall time limit:
#SBATCH --time=7-0:0:0
# Other parameters:

#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1 --cpus-per-task=10

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module load mpi4py/3.1.4-gompi-2023a
source ~/tdhf_env/bin/activate

{run_command}
    """
    return jobscript
import os
jobscript_names=[]
for molecule in molecule_names:
    if molecule=="LiH":
        initlen=20
        n_extra=4
        type="m" #for monomer
    if molecule=="LiH2":
        initlen=34
        n_extra=4
        type="d" #for dimer
    num_gauss=initlen+n_extra

    for fieldstrength in fieldstrengths:
        if molecule=="LiH" and fieldstrength==1:
            max_res=[0.4,1.1,5]
        elif molecule=="LiH2" and fieldstrength==1:
            max_res=[1,3,10]
        for method in methods:
            for rothe_epsilon in max_res:
                E0=math.sqrt(fieldstrength/(3.50944758*1e2))

                outfilefilename="WF_%s_%s_%.4f_%d_%d_%d_%.3e.npz"%(method,molecule,E0,initlen,num_gauss,maxiter,rothe_epsilon)
                if not os.path.exists(outfilefilename):
                    t_init=0
                else:
                    t_init=1000 #arbitrary large number larger than tmax
                p=[initlen,n_extra,fieldstrength,maxiter,t_init,molecule,rothe_epsilon,method,method,molecule,fieldstrength,rothe_epsilon]
                run_command = f"srun --cpus-per-task=10 python -u gauss_Rothe.py {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} freeze {p[6]} False {p[7]} > outputs/{p[8]}_{p[9]}_{p[10]}_{p[11]}.txt"
                jobscript_name=f"{p[8]}_{type}_{p[10]}_{p[11]}"
                jobscript=make_jobscript(jobscript_name,run_command)
                
                with open(f"jobscript_{jobscript_name}.sh","w") as f:
                    f.write(jobscript)
                    print(f"Written {jobscript_name}.sh")
                    jobscript_names.append(f"jobscript_{jobscript_name}.sh")

batch_script_name = "submit_all_jobs.sh"
with open(batch_script_name, "w") as batch_script:
    batch_script.write("#!/bin/bash\n")
    for jobscript_name in jobscript_names:
        batch_script.write(f"sbatch {jobscript_name}\n")
print(f"Written {batch_script_name}")
