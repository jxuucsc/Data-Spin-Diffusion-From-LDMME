#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
sys.path.insert(1, '/home/xujq/codes/jdftx/tools/solve_linearDMME/')
from electron import *
from electron_phonon import *
import solve_gevp_rgat as rgat
from help_solve_tl_fromldmme_ldbdfiles import *
from relax_realtime import *
from diffuse_realspace import *
import solve_tl_fromldmme_RayleighRitz as tlRR
import units
from timeit import default_timer as timer

def solve_diffusion_lengths_linalg_driver(typeEq, e, UL, UR, eph, elec, write_eigen, save_LpinvLv):
  if typeEq == "Lpinv 1":
    return solve_diffusion_lengths_linalg_Lpinv(1, e, UL, UR, eph, elec, write_eigen, save_LpinvLv)
  elif typeEq == "Lpinv 2":
    return solve_diffusion_lengths_linalg_Lpinv(2, e, UL, UR, eph, elec, write_eigen, save_LpinvLv)
  elif typeEq == "Lvtilde":
    return solve_diffusion_lengths_linalg_Lvtilde(e, UL, UR, eph, elec, write_eigen, save_LpinvLv)
  elif typeEq == "gevp":
    return solve_diffusion_lengths_gevp_linalg_eig(e, UL, UR, eph, elec, write_eigen)
  else:
    print(typeEq+" is not an allowed value of typeEq")
    exit(1)

def solve_diffusion_lengths_linalg_Lpinv(typeLpinv, e, UL, UR, eph, elec, write_eigen, save_LpinvLv):
  print("\n--------------------------------------------------")
  print("solving -L^(-1) Lv x = x lambda^(-1)")
  print("--------------------------------------------------")
  t0 = timer()
  stype = "_Lpinv1"
  
  Lpinv = np.array([])
  if typeLpinv == 1:
    einv = np.where(np.real(e) != 0, 1./e, 0)
    einv[0] = 0
    Lpinv = np.einsum("am,bm->ab", np.einsum("am,m->am", UR, -einv, optimize=True), UL.conj(), optimize=True)
  else:
    Lpinv = sci.linalg.pinv(eph.L)
  
  A_E = -np.einsum("a,a->", elec.DfDk.conj(), np.einsum("ab,b->a", Lpinv, elec.DfDk))
  B_E = np.dot(elec.DfDk.conj(), elec.DfDk)
  rhoeqE1 = -compute_rhoeqE1(Lpinv, elec.DfDk)
  compute_carrier_transport_ldmme(rhoeqE1, elec, elec.latt)
  print("effective lifetime in ps: ", units.ps*np.real(A_E/B_E))
  del rhoeqE1
  
  lambda_l  = np.array([])
  UL_l = np.array([])
  UR_l = np.array([])
  Lpinv_Lv = np.array([])
  read_eigen = os.path.exists("restart_ldmme/lambda_diffuse"+stype+".bin") and os.path.exists("restart_ldmme/UL_diffuse"+stype+".bin") and os.path.exists("restart_ldmme/UR_diffuse"+stype+".bin")
  if not read_eigen:
    Lpinv_Lv = np.einsum("ab,bc->ac", Lpinv, elec.Lv, optimize=True)
    t2 = timer()
    E_l, UL_l, UR_l = sci.linalg.eig(-Lpinv_Lv, left=True)
    t3 = timer()
    print("\ntime for sci.linalg.eig: ",t3-t2)
    lambda_l = 1./E_l
  else:
    lambda_l = np.fromfile("restart_ldmme/lambda_diffuse"+stype+".bin", np.complex128).reshape(elec.nmode)
    UL_l = np.fromfile("restart_ldmme/UL_diffuse"+stype+".bin", np.complex128).reshape(elec.nmode,elec.nmode)
    UR_l = np.fromfile("restart_ldmme/UR_diffuse"+stype+".bin", np.complex128).reshape(elec.nmode,elec.nmode)
    if save_LpinvLv:
      Lpinv_Lv = np.fromfile("restart_ldmme/Lpinv_Lv.bin", np.complex128).reshape(elec.nmode,elec.nmode)
  ldiff,lambda_l,UL_l,UR_l = reorder_modes(lambda_l, UL_l, UR_l)
  print("spin diffusion length using pinv of L in micro-meter:\n", ldiff[0:20]*0.529177e-4)
  print("precession freq. in au:\n", lambda_l[0:20].imag)
  print("short diffusion length using pinv of L in micro-meter:\n", ldiff[-21:-1]*0.529177e-4)
  
  # by our construction, U^{L,H} U^R = I
  UL_l, UR_l = re_ortho_normalize_ULUR(UL_l, UR_l)
  if write_eigen and not read_eigen:
    lambda_l.tofile("restart_ldmme/lambda_diffuse"+stype+".bin")
    UL_l.tofile("restart_ldmme/UL_diffuse"+stype+".bin")
    UR_l.tofile("restart_ldmme/UR_diffuse"+stype+".bin")
    if save_LpinvLv:
      Lpinv_Lv.tofile("restart_ldmme/Lpinv_Lv.bin")
  
  if save_LpinvLv:
    eph.Lpinv_Lv = Lpinv_Lv
  t1 = timer()
  print("\ntime for solving diffusion lengths via lin. alg.: ",t1-t0)
  sys.stdout.flush()
  return ldiff,lambda_l,UL_l,UR_l

def solve_diffusion_lengths_linalg_Lvtilde(e, UL, UR, elec, write_eigen, save_LpinvLv):
  print("\n--------------------------------------------------")
  print("solving E_L^(-1) (UL_L^H Lv UR_L) x = x lambda^(-1)")
  print("--------------------------------------------------")
  t0 = timer()
  stype = "_Lvtilde"
  
  Lvtilde = np.einsum("ba,bc->ac", UL[:,1:].conj(), np.einsum("ab,bc->ac", elec.Lv, UR[:,1:], optimize=True), optimize=True)
  einv_Lvtilde = np.einsum("a,ab->ab", 1./e[1:], Lvtilde, optimize=True)
  if save_LpinvLv:
    eph.Lpinv_Lv = np.einsum("am,bm->ab", np.einsum("am,mn->an", UR[:,1:], -einv_Lvtilde, optimize=True), UL[:,1:].conj(), optimize=True)
  
  lambda_l  = np.array([])
  UL_l = np.array([])
  UR_l = np.array([])
  read_eigen = os.path.exists("restart_ldmme/lambda_diffuse"+stype+".bin") and os.path.exists("restart_ldmme/UL_diffuse"+stype+".bin") and os.path.exists("restart_ldmme/UR_diffuse"+stype+".bin")
  if not read_eigen:
    t2 = timer()
    E_l, UL_l, UR_l = sci.linalg.eig(einv_Lvtilde, left=True)
    t3 = timer()
    print("\ntime for sci.linalg.eig: ",t3-t2)
    lambda_l = 1./E_l
    UL_l = np.einsum("ab,bc->ac", UL[:,1:], UL_l, optimize=True)
    UR_l = np.einsum("ab,bc->ac", UR[:,1:], UR_l, optimize=True)
  else:
    lambda_l = np.fromfile("restart_ldmme/lambda_diffuse"+stype+".bin", np.complex128).reshape(elec.nmode-1)
    UL_l = np.fromfile("restart_ldmme/UL_diffuse"+stype+".bin", np.complex128).reshape(elec.nmode,elec.nmode-1)
    UR_l = np.fromfile("restart_ldmme/UR_diffuse"+stype+".bin", np.complex128).reshape(elec.nmode,elec.nmode-1)
  ldiff,lambda_l,UL_l,UR_l = reorder_modes(lambda_l, UL_l, UR_l)
  print("spin diffusion length using pinv of L in micro-meter:\n", ldiff[0:20]*0.529177e-4)
  print("precession freq. in au:\n", lambda_l[0:20].imag)
  print("short diffusion length using pinv of L in micro-meter:\n", ldiff[-21:-1]*0.529177e-4)
  
  # by our construction, U^{L,H} U^R = I
  UL_l, UR_l = re_ortho_normalize_ULUR(UL_l, UR_l)
  if write_eigen and not read_eigen:
    lambda_l.tofile("restart_ldmme/lambda_diffuse"+stype+".bin")
    UL_l.tofile("restart_ldmme/UL_diffuse"+stype+".bin")
    UR_l.tofile("restart_ldmme/UR_diffuse"+stype+".bin")
  
  t1 = timer()
  print("\ntime for solving diffusion lengths via lin. alg.: ",t1-t0)
  sys.stdout.flush()
  return ldiff,lambda_l,UL_l,UR_l

def solve_diffusion_lengths_gevp_linalg_eig(e, UL, UR, eph, elec, write_eigen):
  print("\n--------------------------------------------------")
  print("solving Lv x = L x lambda^(-1)")
  print("--------------------------------------------------")
  t0 = timer()
  stype = "_gevp"
    
  lambda_l  = np.array([])
  UL_l = np.array([])
  UR_l = np.array([])
  read_eigen = os.path.exists("restart_ldmme/lambda_diffuse"+stype+".bin") and os.path.exists("restart_ldmme/UL_diffuse"+stype+".bin") and os.path.exists("restart_ldmme/UR_diffuse"+stype+".bin")
  if not read_eigen:
    t2 = timer()
    print("Lv shape: ",elec.Lv.shape)
    print("eph.L shape: ",eph.L.shape)
    E_l, UL_l, UR_l = sci.linalg.eig(elec.Lv, b=-eph.L, left=True)
    UL_l = np.einsum("ab,bc->ac", L.conj().T, UL_l)
    t3 = timer()
    print("\ntime for sci.linalg.eig: ",t3-t2)
    lambda_l = 1./E_l
  else:
    lambda_l = np.fromfile("restart_ldmme/lambda_diffuse"+stype+".bin", np.complex128).reshape(elec.nmode)
    UL_l = np.fromfile("restart_ldmme/UL_diffuse"+stype+".bin", np.complex128).reshape(elec.nmode,elec.nmode)
    UR_l = np.fromfile("restart_ldmme/UR_diffuse"+stype+".bin", np.complex128).reshape(elec.nmode,elec.nmode)
  ldiff,lambda_l,UL_l,UR_l = reorder_modes(lambda_l, UL_l, UR_l)
  print("spin diffusion length using pinv of L in micro-meter:\n", ldiff[0:20]*0.529177e-4)
  print("precession freq. in au:\n", lambda_l[0:20].imag)
  print("short diffusion length using pinv of L in micro-meter:\n", ldiff[-21:-1]*0.529177e-4)
  
  # check if UL and UR are orthogonal
  o = np.einsum("ab,bc->ac", UL_l.T.conj(), UR_l, optimize=True)
  err = o - np.diag(np.diag(o))
  
  # by our construction, U^{L,H} U^R = I
  print("error of o-o^d: ", np.amax(np.abs(err)))
  
  UL_l, UR_l = re_ortho_normalize_ULUR(UL_l, UR_l)
  if write_eigen and not read_eigen:
    lambda_l.tofile("restart_ldmme/lambda_diffuse"+stype+".bin")
    UL_l.tofile("restart_ldmme/UL_diffuse"+stype+".bin")
    UR_l.tofile("restart_ldmme/UR_diffuse"+stype+".bin")
  
  # check U
  err = np.einsum("ab,bc->ac", UL_l.T.conj(), UR_l, optimize=True) - np.eye(UR_l.shape[0])
  print("error of o-I: ", np.amax(np.abs(err)))
  err = np.einsum("ab,bc->ac", elec.Lv, UR_l, optimize=True) - np.einsum("ab,bc,c->ac", -eph.L, UR_l, 1./lambda_l, optimize=True)
  print("error of Lv UR - (-L) UR lambda^-1: ", np.amax(np.abs(err)))
  err = np.einsum("ab,bc->ac", UL_l.T.conj(), elec.Lv, optimize=True) - np.einsum("a,ab,bc->ac", 1./lambda_l, UL_l.T.conj(), -eph.L, optimize=True)
  print("error of UL^H Lv - lambda^-1 UL^H (-L): ", np.amax(np.abs(err)))
  
  t1 = timer()
  print("\ntime for solving diffusion lengths via lin. alg.: ",t1-t0)
  sys.stdout.flush()
  return ldiff,lambda_l,UL_l,UR_l

def setU_RayleighRitz_forLength(orthnorm, UL, UR, L, Lv, n_pert, nmode, if_norm = True):
  UR_l_RR = np.zeros((nmode,8*n_pert), dtype=np.complex128)
  UR_l_RR[:,0:n_pert]          =  UR[:,1:1+n_pert].copy()
  UR_l_RR[:,n_pert:2*n_pert]   =  np.einsum("ab,bm->am", Lv, UR_l_RR[:,0:n_pert])
  UR_l_RR[:,2*n_pert:3*n_pert] = -np.einsum("ab,bm->am", L,  UR_l_RR[:,n_pert:2*n_pert])
  UR_l_RR[:,3*n_pert:4*n_pert] =  np.einsum("ab,bm->am", Lv, UR_l_RR[:,n_pert:2*n_pert])
  UR_l_RR[:,4*n_pert:5*n_pert] = -np.einsum("ab,bm->am", L,  UR_l_RR[:,2*n_pert:3*n_pert])
  UR_l_RR[:,5*n_pert:6*n_pert] = -np.einsum("ab,bm->am", L,  UR_l_RR[:,3*n_pert:4*n_pert])
  UR_l_RR[:,6*n_pert:7*n_pert] =  np.einsum("ab,bm->am", Lv, UR_l_RR[:,2*n_pert:3*n_pert])
  UR_l_RR[:,7*n_pert:8*n_pert] =  np.einsum("ab,bm->am", Lv, UR_l_RR[:,3*n_pert:4*n_pert])
  UR_l_RR[:,0:8*n_pert] = rgat.proj_out(UR_l_RR[:,0:8*n_pert], UR[:,0:1])
  if if_norm:
    UR_l_RR[:,0:8*n_pert] = np.transpose(normalize_vector(np.transpose(UR_l_RR[:,0:8*n_pert])))
    if orthnorm:
      UR_l_RR,_ = np.linalg.qr(UR_l_RR) # QR factorization to obtain orthonormal basis
  
  UL_l_RR = np.zeros((nmode,8*n_pert), dtype=np.complex128)
  UL_l_RR[:,0:n_pert]          =  UL[:,1:1+n_pert].copy()
  UL_l_RR[:,n_pert:2*n_pert]   =  np.einsum("ba,bm->am", Lv.conj(), UL_l_RR[:,0:n_pert])
  UL_l_RR[:,2*n_pert:3*n_pert] = -np.einsum("ba,bm->am", L.conj(),  UL_l_RR[:,n_pert:2*n_pert])
  UL_l_RR[:,3*n_pert:4*n_pert] =  np.einsum("ba,bm->am", Lv.conj(), UL_l_RR[:,n_pert:2*n_pert])
  UL_l_RR[:,4*n_pert:5*n_pert] = -np.einsum("ba,bm->am", L.conj(),  UL_l_RR[:,2*n_pert:3*n_pert])
  UL_l_RR[:,5*n_pert:6*n_pert] = -np.einsum("ba,bm->am", L.conj(),  UL_l_RR[:,3*n_pert:4*n_pert])
  UL_l_RR[:,6*n_pert:7*n_pert] =  np.einsum("ba,bm->am", Lv.conj(), UL_l_RR[:,2*n_pert:3*n_pert])
  UL_l_RR[:,7*n_pert:8*n_pert] =  np.einsum("ba,bm->am", Lv.conj(), UL_l_RR[:,3*n_pert:4*n_pert])
  UL_l_RR[:,0:8*n_pert] = rgat.proj_out(UL_l_RR[:,0:8*n_pert], UL[:,0:1])
  if if_norm:
    UL_l_RR[:,0:8*n_pert] = np.transpose(normalize_vector(np.transpose(UL_l_RR[:,0:8*n_pert])))
    if orthnorm:
      UL_l_RR,_ = np.linalg.qr(UL_l_RR) # QR factorization to obtain orthonormal basis
  return UL_l_RR, UR_l_RR