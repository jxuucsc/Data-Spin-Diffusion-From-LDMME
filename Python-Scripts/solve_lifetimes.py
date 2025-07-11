#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
sys.path.insert(1, '/home/xujq/codes/jdftx/tools/solve_linearDMME/')
import solve_gevp_rgat as rgat
from help_solve_tl_fromldmme_ldbdfiles import *
from relax_realtime import *
from diffuse_realspace import *
import solve_tl_fromldmme_RayleighRitz as tlRR
import units
from timeit import default_timer as timer

def solve_lifetimes_linalg(L, comment, nmode, write_eigen):
  print("\n--------------------------------------------------")
  print("solve "+comment+" :")
  print("--------------------------------------------------")
  t0 = timer()
  
  e = np.array([])
  UL = np.array([])
  UR = np.array([])
  read_eigen = os.path.exists("restart_ldmme/e_relax.bin") and os.path.exists("restart_ldmme/UL_relax.bin") and os.path.exists("restart_ldmme/UR_relax.bin")
  if not read_eigen:
    e, UL, UR = sci.linalg.eig(-L, left=True)
  else:
    e  = np.fromfile("restart_ldmme/e_relax.bin",  np.complex128).reshape(nmode)
    UL = np.fromfile("restart_ldmme/UL_relax.bin", np.complex128).reshape(nmode,nmode)
    UR = np.fromfile("restart_ldmme/UR_relax.bin", np.complex128).reshape(nmode,nmode)
  tau,e,UL,UR = reorder_modes(e, UL, UR)
  print("rates:\n", e[0:10])
  print("spin lifetimes in ps:\n", units.ps*tau[0:10])
  print("precession freq. in au:\n", e[0:10].imag)
  print("short lifetimes in ps:\n", units.ps*tau[-11:-1])
  
  # by our construction, U^{L,H} U^R = I
  UL, UR = re_ortho_normalize_ULUR(UL, UR)
  if write_eigen and not read_eigen:
    e.tofile("restart_ldmme/e_relax.bin")
    UL.tofile("restart_ldmme/UL_relax.bin")
    UR.tofile("restart_ldmme/UR_relax.bin")
  
  t1 = timer()
  print("\ntime for solving lifetimes via lin. alg.: ",t1-t0)
  sys.stdout.flush()
  return tau,e,UL,UR

def setU_RayleighRitz_forLifetime(orthnorm, ob, rhopert, L, order, mask_rho, nk, nb):
  n_pert = rhopert.shape[0]
  UR_RR = np.zeros((ob.shape[1],1+(n_pert-1)*order), dtype=np.complex128)
  UL_RR = np.zeros((ob.shape[1],1+(n_pert-1)*order), dtype=np.complex128)
  UR_RR[:,0:n_pert] = np.transpose(rhopert)
  UL_RR[:,0:n_pert] = np.transpose(normalize_vector(ob[0:n_pert]))
  UL_RR[:,0:n_pert], UR_RR[:,0:n_pert] = re_ortho_normalize_ULUR(UL_RR[:,0:n_pert], UR_RR[:,0:n_pert], True)
  for io in range(1,order):
    UR_RR[:,1+(n_pert-1)*io:n_pert+(n_pert-1)*io] = -np.einsum("ab,bm->am", L,        UR_RR[:,-(n_pert-2)+(n_pert-1)*io:1+(n_pert-1)*io])
    UL_RR[:,1+(n_pert-1)*io:n_pert+(n_pert-1)*io] = -np.einsum("ba,bm->am", L.conj(), UL_RR[:,-(n_pert-2)+(n_pert-1)*io:1+(n_pert-1)*io])
    UR_RR[:,1+(n_pert-1)*io:n_pert+(n_pert-1)*io] = np.transpose(normalize_vector(np.transpose(UR_RR[:,1+(n_pert-1)*io:n_pert+(n_pert-1)*io])))
    UL_RR[:,1+(n_pert-1)*io:n_pert+(n_pert-1)*io] = np.transpose(normalize_vector(np.transpose(UL_RR[:,1+(n_pert-1)*io:n_pert+(n_pert-1)*io])))
  if orthnorm:
    UR_RR,_ = np.linalg.qr(UR_RR)
    UL_RR,_ = np.linalg.qr(UL_RR)
    make_U_hermitian(UR_RR, nk, nb, mask_rho)
    make_U_hermitian(UL_RR, nk, nb, mask_rho)
  return UL_RR, UR_RR