#!/bin/bash
#SBATCH -p windfall
#SBATCH --account=windfall
##SBATCH -p cpuq
##SBATCH --account=cpuq
#SBATCH -N 8
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=40
#SBATCH -J jdftx

module load myopenmpi-4.0.7_gcc gsl

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/data/groups/ping/jxu153/codes/jdftx/jdftx-202306/build-test2"
DIRF="/data/groups/ping/jxu153/codes/jdftx/jdftx-202306/build-FeynWann-test3"

${MPICMD} ${DIRF}/lindbladInit_for-DMD-4.5.6/init_for-DMD -i lindbladInit.in > lindbladInit.out
