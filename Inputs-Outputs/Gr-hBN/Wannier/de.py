#!/usr/bin/env python
import numpy as np

c1 = np.loadtxt("wannier.eigenvals", dtype=np.float64, usecols=(12))
c2 = np.loadtxt("wannier.eigenvals", dtype=np.float64, usecols=(13))
dc = c2 - c1
print("from k130 to k150 around CBM\n",dc[437])
