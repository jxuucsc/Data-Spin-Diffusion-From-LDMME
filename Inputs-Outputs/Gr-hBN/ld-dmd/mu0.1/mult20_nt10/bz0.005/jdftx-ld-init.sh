#!/bin/bash
#SBATCH -p qt
#SBATCH -N 2
#SBATCH -t 99:00:00
#SBATCH --ntasks-per-node=56
#SBATCH -J jdftx

module load gcc/8.5.0 mkl mpi

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/home/xujq/codes/jdftx/jdftx-202404/build"
DIRF="/home/xujq/codes/jdftx/jdftx-202404/build-FeynWann"

${MPICMD} ${DIRF}/lindbladInit_for-DMD-4.5.6/init_for-DMD -i lindbladInit.in > lindbladInit.out
