#!/bin/bash
#SBATCH -p gpuq
#SBATCH --account=gpuq
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -J fit

./fitQuality.py > fitQuality.out
