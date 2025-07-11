#!/usr/bin/gnuplot -persist
set xtics ( "Gamma" 0,  "M" 250,  "K" 437,  "Gamma" 673 )
set key center center font ",15"
nRows = real(system("awk '$1==\"kpoint\" {nRows++} END {print nRows}' bandstruct.kpoints"))
nCols = real(system("wc -c < bandstruct.eigenvals")) / (8*nRows)
formatString = system(sprintf("echo '' | awk 'END { str=\"\"; for(i=0; i<%d; i++) str = str \"%%\" \"lf\"; print str}'", nCols))
VBM=-0.162021
ha2ev=27.211386
set xzeroaxis               #Add dotted line at zero energy
set ylabel "E - mu (eV)"   #Add y-axis label
set term png
set output "bandstruct_full.png"
set yrange [-20:15]
plot "bandstruct.eigenvals" binary format=formatString u 0:((column(1)-VBM)*ha2ev) t "DFT" w p ps 2 pt 2 lc rgb "black", \
     "wannier.eigenvals" u (1.*column(0)):((column(1)-VBM)*ha2ev) t "Wannier" w l lw 2 lc rgb "red", \
     for [i=2:nCols] "bandstruct.eigenvals" binary format=formatString u 0:((column(i)-VBM)*ha2ev) t "" w p ps 0.5 pt 2 lc rgb "black", \
     for [i=2:30] "wannier.eigenvals" u (1.*column(0)):((column(i)-VBM)*ha2ev) t "" w l lw 2 lc rgb "red"
