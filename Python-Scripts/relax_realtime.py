#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
import units
from timeit import default_timer as timer

def relax_realtime(t, rhopert, ob, UL, UR, e, what, thr = 0):
  rhopert_relevance = np.einsum("am,a->m", UL.conj(), rhopert)
  ob_relevance = np.einsum("a,am->m", ob.conj(), UR)
  relevance = rhopert_relevance * ob_relevance
  mask_relevance = np.abs(relevance) > thr * np.max(np.abs(relevance))
  relevance = relevance[mask_relevance]
  e = e[mask_relevance]
  print(e.shape[0]," (of ",UR.shape[1],") modes are considered in real-time simulation")
  expt = np.exp(np.outer(-e, t))
  Ot = np.real(np.einsum("m,mt->t", relevance, expt))
  np.savetxt(what+".dat", np.transpose([t*units.ps, Ot, np.abs(Ot/Ot[0])]), comments='# t (ps), Observable')
  return Ot