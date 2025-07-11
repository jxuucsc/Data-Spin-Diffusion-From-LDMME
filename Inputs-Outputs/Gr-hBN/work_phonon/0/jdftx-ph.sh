#!/bin/bash
#SBATCH -p qt
#SBATCH -N 1
#SBATCH -t 99:00:00
#SBATCH --ntasks-per-node=XX1
#SBATCH -J jdftx

echo $SLURM_NODELIST
module load gcc/8.5.0 mkl mpi

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/home/xujq/codes/jdftx/jdftx-202306/build"
DIRF="/home/xujq/codes/jdftx/jdftx-202306/build-FeynWann"

i=XX2
export phononParams="iPerturbation $i"
#${MPICMD} ${DIRJ}/phonon -i phonon-scf.in > phonon-scf.${i}.out
${MPICMD} ${DIRJ}/phonon -i phonon.in > phonon.${i}.out
