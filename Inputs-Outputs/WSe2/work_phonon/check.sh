#!/bin/bash
for i in {1..7}; do
cd pert$i
echo $i >> ../ftmp
grep "ElecMinimize: Converged (|Delta Etot|<1.000000e-" phonon.*.out >> ../ftmp
cd ..
done
