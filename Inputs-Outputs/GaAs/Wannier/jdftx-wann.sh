#!/bin/bash
#SBATCH -p windfall
#SBATCH --account=windfall
##SBATCH -p gpuq
##SBATCH --account=gpuq
#SBATCH -N 2
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=40
#SBATCH -J jdftx

module load python/3.8.6
module load myopenmpi-4.0.7_gcc gsl

MPICMD="mpirun -np $SLURM_NTASKS"
DIRJ="/data/groups/ping/jxu153/codes/jdftx/jdftx-202306/build-test2"
DIRF="/data/groups/ping/jxu153/codes/jdftx/jdftx-202306/build-FeynWann-test3"

prfx=wannier
#${MPICMD} ${DIRJ}/wannier -i ${prfx}.in > ${prfx}.out
#python3 fitQuality.py > fitQuality.out
#exit 0
python3 rand_wann-centers.py
cp wannier.in0 wannier.in
cat rand_wann-centers.dat >> wannier.in
rm rand_wann-centers.dat
${MPICMD} ${DIRJ}/wannier -i ${prfx}.in > ${prfx}.out
python3 fitQuality.py > fitQuality.out
