#!/bin/bash
#SBATCH -p cpuq
#SBATCH --account=cpuq
#SBATCH -N 8
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=8
#SBATCH -J jdftx

module load myopenmpi-4.0.7_gcc gsl intel/mkl

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/data/groups/ping/jxu153/codes/jdftx/jdftx-202306/build-test2"
DIRF="/data/groups/ping/jxu153/codes/jdftx/jdftx-202306/build-FeynWann-test3"

${MPICMD} ${DIRJ}/jdftx -i scf.in > scf.out
${MPICMD} ${DIRJ}/jdftx -i totalE.in > totalE.out
#${MPICMD} ${DIRJ}/phonon -ni phonon.in > split.out
${MPICMD} ${DIRJ}/jdftx -i bandstruct.in > bandstruct.out
