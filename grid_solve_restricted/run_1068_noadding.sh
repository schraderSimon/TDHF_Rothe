#!/bin/bash
# Job name:
#SBATCH --job-name=noadding
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

srun --exclusive -N1 -n1 --cpus-per-task=4 bash -c 'export OMP_NUM_THREADS=4; python -u gauss_Rothe.py 4.0 300 0 LiH freeze 100000 False DFT' \
     > outputs/DFT_LiH_100000_300.txt &
srun --exclusive -N1 -n1 --cpus-per-task=4 bash -c 'export OMP_NUM_THREADS=4; python -u gauss_Rothe.py 4.0 0 0 LiH freeze 100000 False DFT' \
     > outputs/DFT_LiH_100000_0.txt &
srun --exclusive -N1 -n1 --cpus-per-task=4 bash -c 'export OMP_NUM_THREADS=4; python -u gauss_Rothe.py 4.0 300 0 LiH freeze 100000 False HF' \
     > outputs/HF_LiH_100000_300.txt &
srun --exclusive -N1 -n1 --cpus-per-task=4 bash -c 'export OMP_NUM_THREADS=4; python -u gauss_Rothe.py 4.0 0 0 LiH freeze 100000 False HF' \
     > outputs/HF_LiH_100000_0.txt &
srun --exclusive -N1 -n1 --cpus-per-task=4 bash -c 'export OMP_NUM_THREADS=4; python -u gauss_Rothe.py 4.0 300 0 LiH2 freeze 100000 False DFT' \
     > outputs/DFT_LiH2_100000_300.txt &
srun --exclusive -N1 -n1 --cpus-per-task=4 bash -c 'export OMP_NUM_THREADS=4; python -u gauss_Rothe.py 4.0 0 0 LiH2 freeze 100000 False DFT' \
     > outputs/DFT_LiH2_100000_0.txt &
srun --exclusive -N1 -n1 --cpus-per-task=4 bash -c 'export OMP_NUM_THREADS=4; python -u gauss_Rothe.py 4.0 300 0 LiH2 freeze 100000 False HF' \
     > outputs/HF_LiH2_100000_300.txt &
srun --exclusive -N1 -n1 --cpus-per-task=4 bash -c 'export OMP_NUM_THREADS=4; python -u gauss_Rothe.py 4.0 0 0 LiH2 freeze 100000 False HF' \
     > outputs/HF_LiH2_100000_0.txt &
wait
