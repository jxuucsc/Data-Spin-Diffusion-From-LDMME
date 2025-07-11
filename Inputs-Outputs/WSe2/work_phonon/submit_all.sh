#!/bin/bash
for i in {1..7}; do
cd pert$i
sbatch jdftx-ph.sh
cd ..
done
