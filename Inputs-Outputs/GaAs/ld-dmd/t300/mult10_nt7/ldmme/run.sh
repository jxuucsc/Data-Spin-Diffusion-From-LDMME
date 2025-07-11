#!/bin/bash
##SBATCH -p windfall
##SBATCH --account=windfall
#SBATCH -p cpuq
#SBATCH --account=cpuq
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -J jdftx

module load python/3.8.6

python3 solve_tl_fromldmme_ldbdfiles_gaas.py > solve_tl_fromldmme_ldbdfiles_gaas.out
