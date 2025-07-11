#!/bin/bash
#SBATCH -p qt
#SBATCH -N 2
#SBATCH -t 99:00:00
#SBATCH --ntasks-per-node=14
#SBATCH -J jdftx

echo $SLURM_NODELIST
module load gcc/8.5.0 mkl mpi

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/home/xujq/codes/jdftx/jdftx-202404/build"
DIRF="/home/xujq/codes/jdftx/jdftx-202404/build-FeynWann"

${MPICMD} ${DIRJ}/jdftx -i scf.in > scf.out
${MPICMD} ${DIRJ}/jdftx -i totalE.in > totalE.out
#${MPICMD} ${DIRJ}/phonon -ni phonon.in > split.out
${MPICMD} ${DIRJ}/jdftx -i bandstruct.in > bandstruct.out
