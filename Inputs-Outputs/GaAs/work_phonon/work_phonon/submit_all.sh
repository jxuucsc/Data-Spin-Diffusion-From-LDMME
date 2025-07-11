#!/bin/bash
for i in {1..4}; do
cd pert$i
sbatch jdftx-ph.sh
cd ..
done
