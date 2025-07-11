#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
sys.path.insert(1, '/home/xujq/codes/jdftx/tools/solve_linearDMME/')
import solve_gevp_rgat as rgat
from help_solve_tl_fromldmme_ldbdfiles import *
import units
from timeit import default_timer as timer

def rayleigh_ritz_method(A, B, VL, VR, type_e = "smallest real abs", opt_einsum = True):
  Atilde = np.einsum("ba,bc->ac", VL.conj(), np.einsum("ab,bc->ac", A, VR, optimize=opt_einsum), optimize=opt_einsum)
  Btilde = np.einsum("ba,bc->ac", VL.conj(), np.einsum("ab,bc->ac", B, VR, optimize=opt_einsum), optimize=opt_einsum)
  E, YR = sci.linalg.eig(Atilde, b=Btilde)
  return rgat.sort_E_Y(type_e, E, YR)

def rayleigh_ritz_method_with_left(A, B, VL, VR, type_e = "smallest real abs", opt_einsum = True):
  Atilde = np.einsum("ba,bc->ac", VL.conj(), np.einsum("ab,bc->ac", A, VR, optimize=opt_einsum), optimize=opt_einsum)
  Btilde = np.einsum("ba,bc->ac", VL.conj(), np.einsum("ab,bc->ac", B, VR, optimize=opt_einsum), optimize=opt_einsum)
  E, YL, YR = sci.linalg.eig(Atilde, b=Btilde, left=True)
  YL = np.einsum("ab,bc->ac", Btilde.conj().T, YL)
  return rgat.sort_E_Y(type_e, E, YR, YL)

def rayleigh_ritz_method_with_left_g2s(A, B, VL, VR, type_e = "smallest real abs", opt_einsum = True):
  Atilde = np.einsum("ba,bc->ac", VL.conj(), np.einsum("ab,bc->ac", A, VR, optimize=opt_einsum), optimize=opt_einsum)
  Btilde = np.einsum("ba,bc->ac", VL.conj(), np.einsum("ab,bc->ac", B, VR, optimize=opt_einsum), optimize=opt_einsum)
  Atilde = np.einsum("ab,bc->ac", sci.linalg.inv(Btilde), Atilde)
  E, YL, YR = sci.linalg.eig(Atilde, left=True)
  return rgat.sort_E_Y(type_e, E, YR, YL)

def solve_tl_fromldmme_RayleighRitz(typeE, A, B_, VL_full, VR_full, indV, commV, prt_imag, type_e = "smallest real abs", opt_einsum = True):
  B = np.eye(A.shape[0], dtype=np.complex128)
  if B_ is not None:
    B = B_
  VL = VL_full[:,indV]
  VR = VR_full[:,indV]
  E, Y = rayleigh_ritz_method(A, B, VL, VR, type_e, opt_einsum)
  if typeE == "rate":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") lifetime in ps:\n", units.ps/np.real(E))
    if prt_imag:
      print("precession frequency:", np.imag(E))
  elif typeE == "lambda inv":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") diffusion length in micro-meter:\n", 1e6*units.meter/np.real(1./E))
    if prt_imag:
      print("precession frequency:", np.imag(1./E))
  elif typeE == "lambda":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") diffusion length in micro-meter:\n", 1e6*units.meter/np.real(E))
    if prt_imag:
      print("precession frequency:", np.imag(E))
  print("")

def solve_tl_fromldmme_RayleighRitz_solutions(typeE, A, B_, VL_full, VR_full, indV, commV, prt_imag, type_e = "smallest real abs", opt_einsum = True):
  VL = VL_full[:,indV]
  VR = VR_full[:,indV]
  VL, VR = re_ortho_normalize_ULUR(VL, VR, False) #, "left")
  B = np.eye(A.shape[0], dtype=np.complex128)
  if B_ is not None:
    B = B_
  E, YL, YR = rayleigh_ritz_method_with_left(A, B, VL, VR, type_e, opt_einsum)
  YL, YR = re_ortho_normalize_ULUR(YL, YR, True) #, "left")
  VL = np.einsum("ab,bc->ac", VL, YL, optimize=True)
  VR = np.einsum("ab,bc->ac", VR, YR, optimize=True)
  if typeE == "rate":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") lifetime in ps:\n", units.ps/np.real(E))
    if prt_imag:
      print("precession frequency:", np.imag(E))
    print("")
    return E, VL, VR
  elif typeE == "lambda inv":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") diffusion length in micro-meter:\n", 1e6*units.meter/np.real(1./E))
    if prt_imag:
      print("precession frequency:", np.imag(1./E))
    print("")
    return 1./E, VL, VR # return lambda=1/E
  elif typeE == "lambda":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") diffusion length in micro-meter:\n", 1e6*units.meter/np.real(E))
    if prt_imag:
      print("precession frequency:", np.imag(E))
    print("")
    return E, VL, VR
def solve_tl_fromldmme_RayleighRitz_solutions_g2s(typeE, A, B_, VL_full, VR_full, indV, commV, prt_imag, type_e = "smallest real abs", opt_einsum = True):
  VL = VL_full[:,indV]
  VR = VR_full[:,indV]
  VL, VR = re_ortho_normalize_ULUR(VL, VR, False) #, "left")
  B = np.eye(A.shape[0], dtype=np.complex128)
  if B_ is not None:
    B = B_
  E, YL, YR = rayleigh_ritz_method_with_left_g2s(A, B, VL, VR, type_e, opt_einsum)
  YL, YR = re_ortho_normalize_ULUR(YL, YR, True) #, "left")
  VL = np.einsum("ab,bc->ac", VL, YL, optimize=True)
  VR = np.einsum("ab,bc->ac", VR, YR, optimize=True)
  if typeE == "rate":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") lifetime in ps:\n", units.ps/np.real(E))
    if prt_imag:
      print("precession frequency:", np.imag(E))
    print("")
    return E, VL, VR
  elif typeE == "lambda inv":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") diffusion length in micro-meter:\n", 1e6*units.meter/np.real(1./E))
    if prt_imag:
      print("precession frequency:", np.imag(1./E))
    print("")
    return 1./E, VL, VR # return lambda=1/E
  elif typeE == "lambda":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") diffusion length in micro-meter:\n", 1e6*units.meter/np.real(E))
    if prt_imag:
      print("precession frequency:", np.imag(E))
    print("")
    return E, VL, VR

def rayleigh_ritz_method_sevp(A, VL, VR, type_e = "smallest real abs", opt_einsum = True):
  Atilde = np.einsum("ba,bc->ac", VL.conj(), np.einsum("ab,bc->ac", A, VR, optimize=opt_einsum), optimize=opt_einsum)
  E, YL, YR = sci.linalg.eig(Atilde, left=True)
  return rgat.sort_E_Y(type_e, E, YR, YL)

def solve_tl_fromldmme_RayleighRitz_sevp(typeE, A, VL_full, VR_full, indV, commV, prt_imag, type_e = "smallest real abs", opt_einsum = True):
  VL = VL_full[:,indV]
  VR = VR_full[:,indV]
  VL, VR = re_ortho_normalize_ULUR(VL, VR, False) #, "left")
  E, YL, YR = rayleigh_ritz_method_sevp(A, VL, VR, type_e, opt_einsum)
  YL, YR = re_ortho_normalize_ULUR(YL, YR, True) #, "left")
  VL = np.einsum("ab,bc->ac", VL, YL, optimize=True)
  VR = np.einsum("ab,bc->ac", VR, YR, optimize=True)
  if typeE == "rate":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") lifetime in ps:\n", units.ps/np.real(E))
    if prt_imag:
      print("precession frequency:", np.imag(E))
    print("")
    return E, VL, VR
  elif typeE == "lambda inv":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") diffusion length in micro-meter:\n", 1e6*units.meter/np.real(1./E))
    if prt_imag:
      print("precession frequency:", np.imag(1./E))
    print("")
    return 1./E, VL, VR # return lambda=1/E
  elif typeE == "lambda":
    print("Rayleigh-Ritz (",indV.shape[0]," vec., ",commV,") diffusion length in micro-meter:\n", 1e6*units.meter/np.real(E))
    if prt_imag:
      print("precession frequency:", np.imag(E))
    print("")
    return E, VL, VR