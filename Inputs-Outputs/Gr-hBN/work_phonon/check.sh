#!/bin/bash
for i in {1..16}; do
cd pert$i
echo $i >> ../ftmp
grep "ElecMinimize: Converged (|Delta F|<1.000000e-09" phonon.*.out >> ../ftmp
cd ..
done
