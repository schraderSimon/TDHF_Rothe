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
#SBATCH --mem-per-cpu=2G

# Exit on error and unset variables:
set -o errexit
set -o nounset

module load Python/3.12.3-GCCcore-13.3.0
source ~/tdhf_env_312/bin/activate

method="HF"
molecule="LiH"
fieldstrength=4.0
maxiter=300
starttime=0

epsilon1=1
epsilon2=5
epsilon3=10

cpus1=16
cpus2=10
cpus3=6

outfile1="outputs/${method}_${molecule}_${fieldstrength}_${epsilon1}.txt"
srun --exclusive -N1 -n1 --cpus-per-task=$cpus1 bash -c \
  "export OMP_NUM_THREADS=$cpus1; python -u gauss_Rothe.py $fieldstrength $maxiter $starttime $molecule freeze $epsilon1 False $method" \
  > "$outfile1" &

outfile2="outputs/${method}_${molecule}_${fieldstrength}_${epsilon2}.txt"
srun --exclusive -N1 -n1 --cpus-per-task=$cpus2 bash -c \
  "export OMP_NUM_THREADS=$cpus2; python -u gauss_Rothe.py $fieldstrength $maxiter $starttime $molecule freeze $epsilon2 False $method" \
  > "$outfile2" &

outfile3="outputs/${method}_${molecule}_${fieldstrength}_${epsilon3}.txt"
srun --exclusive -N1 -n1 --cpus-per-task=$cpus3 bash -c \
  "export OMP_NUM_THREADS=$cpus3; python -u gauss_Rothe.py $fieldstrength $maxiter $starttime $molecule freeze $epsilon3 False $method" \
  > "$outfile3" &

wait