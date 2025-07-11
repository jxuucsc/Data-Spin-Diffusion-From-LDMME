#!/bin/bash
#SBATCH -p qt
#SBATCH -N 1
#SBATCH -t 99:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -J ldmme

echo $SLURM_NODELIST

/opt/soft/anaconda3/bin/python3 solve_tl_fromldmme_ldbdfiles_grbn.py > solve_tl_fromldmme_ldbdfiles_grbn.out
