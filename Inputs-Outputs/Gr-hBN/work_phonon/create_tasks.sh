#!/bin/bash
np=16
grep -A $np "Parameter summary for supercell calculations:" ../split.out > ftmp
grep -v "Parameter summary for supercell calculations:" ftmp > ftmp2
mv ftmp2 ftmp

while read -r a b c d; do
cp 0/jdftx-ph.sh pert$i/
rm -r pert$b
cp -r 0 pert$b

n=$(((d + 1) / 4 * 4))
cd pert$b
sed -i "s/XX1/$n/g" jdftx-ph.sh
sed -i "s/XX2/$b/g" jdftx-ph.sh
cd ..
done < ftmp
