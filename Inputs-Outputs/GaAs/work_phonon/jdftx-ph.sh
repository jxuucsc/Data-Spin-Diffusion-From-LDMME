#!/bin/bash
#SBATCH -p cpuq
#SBATCH --account=cpuq
#SBATCH -N 16
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=10
#SBATCH -J jdftx

module load myopenmpi-4.0.7_gcc gsl intel/mkl

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/data/groups/ping/jxu153/codes/jdftx/jdftx-202306/build-test2"
DIRF="/data/groups/ping/jxu153/codes/jdftx/jdftx-202306/build-FeynWann-test3"

i=1
export phononParams="iPerturbation $i"
${MPICMD} ${DIRJ}/phonon -i phonon-scf.in > phonon-scf.${i}.out
${MPICMD} ${DIRJ}/phonon -i phonon.in > phonon.${i}.out
