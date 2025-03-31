#!/bin/bash
# Job name:
#SBATCH --job-name=HF_s_1068
#
# Project:
#SBATCH --account=nn4654k
#
# Wall time limit:
#SBATCH --time=7-0:0:0
#
# Allocate one node with all 32 cores:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Exit on error and unset variables:
set -o errexit
set -o nounset

module load Python/3.12.3-GCCcore-13.3.0
source ~/tdhf_env/bin/activate

srun --exclusive -N1 -n1 --cpus-per-task=16 bash -c 'export OMP_NUM_THREADS=16; python -u gauss_Rothe.py 4.0 300 0 LiH freeze 3 False HF' \
     > outputs/HF_LiH_4_3.txt &
srun --exclusive -N1 -n1 --cpus-per-task=10 bash -c 'export OMP_NUM_THREADS=10; python -u gauss_Rothe.py 4.0 300 0 LiH freeze 5 False HF' \
     > outputs/HF_LiH_4_5.txt & #Give this one a last try?
srun --exclusive -N1 -n1 --cpus-per-task=6 bash -c 'export OMP_NUM_THREADS=6; python -u gauss_Rothe.py 4.0 300 0 LiH freeze 10 False HF' \
     > outputs/HF_LiH_4_10.txt & #Give this one a last try?


wait
