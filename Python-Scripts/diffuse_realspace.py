#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
import units
from timeit import default_timer as timer

def diffuse_realspace_1d(x, positive, rhopert, ob, UL, UR, lambda_x, what, thr = 0, mode_resolved = False):
  rhopert_relevance = np.einsum("am,a->m", UL.conj(), rhopert)
  ob_relevance = np.einsum("a,am->m", ob.conj(), UR)
  relevance = rhopert_relevance * ob_relevance
  mask_relevance = np.abs(relevance) > np.abs(thr) * np.max(np.abs(relevance))
  if positive:
    mask_relevance = np.logical_and(mask_relevance, lambda_x > 0)
  else:
    mask_relevance = np.logical_and(mask_relevance, lambda_x < 0)
  relevance = relevance[mask_relevance]
  lambda_x = lambda_x[mask_relevance]
  print(lambda_x.shape[0]," (of ",UR.shape[1],") modes are considered in real-space simulation")
  expx = np.exp(np.outer(-lambda_x, x))
  Ox = np.real(np.einsum("m,mt->t", relevance, expx))
  np.savetxt(what+".dat", np.transpose([x*1e9*units.meter, Ox, np.abs(Ox/Ox[0])]), comments='# x (nm), Observable') # x in nm
  if mode_resolved:
    nm = np.minimum(relevance.shape[0], 10)
    Oxm = np.real(np.einsum("m,mt->mt", relevance, expx))
    out = np.transpose( np.concatenate( ( [x*1e9*units.meter], Oxm, Oxm/Ox[0], [np.abs(Ox/Ox[0])] ) ) )
    np.savetxt(what+"_mode.dat", out, comments='# x (nm), mode-res. ob., norm. mode-res. ob., norm. tot. ob.') # x in nm
  return Ox