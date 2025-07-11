#!/bin/bash
#SBATCH -p qt
#SBATCH -N 1
#SBATCH -t 99:00:00
#SBATCH --ntasks-per-node=56
#SBATCH -J jdftx

echo $SLURM_NODELIST
module load cmake gcc/8.5.0 mkl mpi

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/home/xujq/codes/jdftx/jdftx-202404/build"
DIRF="/home/xujq/codes/jdftx/jdftx-202404/build-FeynWann"

prfx=wannier
#${MPICMD} ${DIRJ}/wannier -i ${prfx}.in > ${prfx}.out
#/opt/soft/anaconda3/bin/python3 fitQuality_kpts-uniform.py > fitQuality_kpts-uniform.out
#exit 0
/opt/soft/anaconda3/bin/python3 rand_wann-centers.py
cp wannier.in0 wannier.in
cat rand_wann-centers.dat >> wannier.in
rm rand_wann-centers.dat
${MPICMD} ${DIRJ}/wannier -i ${prfx}.in > ${prfx}.out
/opt/soft/anaconda3/bin/python3 fitQuality_kpts-uniform.py > fitQuality_kpts-uniform.out
