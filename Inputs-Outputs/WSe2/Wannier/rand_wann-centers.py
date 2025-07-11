#!/usr/bin/env python
import os
import random

random.seed()

f = open("rand_wann-centers.dat","w")

for i in range(6):
  x = random.random()-0.5
  y = random.random()-0.5
  z = 0.4*(random.random()-0.5)
  f.write("wannier-center Gaussian %10.6f %10.6f %10.6f 2 %s \n" % (x,y,z,"sUp") )
  f.write("wannier-center Gaussian %10.6f %10.6f %10.6f 2 %s \n" % (x,y,z,"sDn") )
  #f.write("wannier-center Gaussian %10.6f %10.6f %10.6f 2 %s \n" % (-x,-y,-z,"sUp") )
  #f.write("wannier-center Gaussian %10.6f %10.6f %10.6f 2 %s \n" % (-x,-y,-z,"sDn") )
