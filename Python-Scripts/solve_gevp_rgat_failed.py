#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
sys.path.insert(1, '/home/xujq/codes/jdftx/tools/solve_linearDMME/')
import units
from timeit import default_timer as timer

def sort_E_Y(type_e, E, YL, YR):
  if type_e == "largest real abs":
    ind = np.argsort(-np.abs(np.real(E)))
  if type_e == "largest real":
    ind = np.argsort(-np.real(E))
  elif type_e == "largest abs":
    ind = np.argsort(-np.abs(E))
  elif type_e == "smallest real +":
    Ep = np.where(np.real(E) <= 1e-15, E[np.real(E).argmax()] + 1, E)
    ind = np.argsort(np.abs(np.real(Ep)))
  elif type_e == "smallest real abs":
    ind = np.argsort(np.abs(np.real(E)))
  elif type_e == "smallest real":
    ind = np.argsort(np.real(E))
  elif type_e == "smallest abs":
    ind = np.argsort(np.abs(E))
  elif type_e == "smallest inv real +":
    Einv = 1./E
    Einvp = np.where(np.real(Einv) <= 1e-15, Einv[np.real(Einv).argmax()] + 1, Einv)
    ind = np.argsort(np.abs(np.real(Einvp)))
  elif type_e == "smallest inv abs":
    Einv = 1./E
    ind = np.argsort(np.abs(Einv))
  elif type_e == "smallest inv real abs":
    Einv = 1./E
    ind = np.argsort(np.abs(np.real(Einv)))
  else:
    print("type_e ",type_e," is not allowed")
    exit(1)
  if YL is None:
    return E[ind], YR[:,ind]
  else:
    return E[ind], YL[:,ind], YR[:,ind]

def compute_error_between(E, E0, type_err = "abs"):
  if type_err == "abs":
    return np.min(np.abs(E - E0))
  elif type_err == "rel":
    return np.min(np.abs(E - E0) / np.abs(E))

def compute_error(type_e, E, E0, type_err = "abs", thr = 1e-10):
  err = 0
  if "largest" in type_e or "smallest inv" in type_e:
    for i in range(E.shape[0]):
      dE_ = 0
      if "real" in type_e:
        dE_ = compute_error_between(np.real(E[i]), np.real(E0[i]), type_err)
      elif "abs" in type_e:
        dE_ = compute_error_between(np.abs(E[i]), np.abs(E0[i]), type_err)
      if dE_ > thr and dE_ > err:
        err = dE_
      else:
        dEabs_min = compute_error_between(E[i], E0, type_err)
        if dEabs_min > err:
          err = dEabs_min
  elif "smallest" in type_e:
    for i in range(E.shape[0]):
      dE_ = 0
      if "real" in type_e:
        dE_ = compute_error_between(np.real(E[i]), np.real(E0[i]), type_err)
      elif "abs" in type_e:
        dE_ = compute_error_between(np.abs(E[i]), np.abs(E0[i]), type_err)
      if dE_ > thr and dE_ > err:
        err = dE_
      else:
        dEabs_min = compute_error_between(E[i], E0, type_err)
        if dEabs_min > err:
          err = dEabs_min
  else:
    print("type_e ",type_e," is not allowed")
    exit(1)
  return err

def check_orthonormal(B, VL, VR, opt_einsum = True):
  if B is None:
    err = np.max(np.abs(np.eye(VR.shape[1], dtype=np.complex128) - np.einsum("ba,bc->ac", VL.conj(), VR, optimize=opt_einsum)))
  else:
    err = np.max(np.abs(np.eye(VR.shape[1], dtype=np.complex128) - np.einsum("ba,bc->ac", VL.conj(), np.einsum("bd,dc->bc", B, VR, optimize=opt_einsum), optimize=opt_einsum)))
  return err

def check_solution(A, B, E, VL, VR, opt_einsum = True):
  if A is None:
    errL = np.max(np.abs(VL - np.einsum("ab,bc->ac", B.T.conj(), np.einsum("bc,c->bc", VL, E.conj(), optimize=opt_einsum), optimize=opt_einsum)))
    errR = np.max(np.abs(VR - np.einsum("ab,bc->ac", B,          np.einsum("bc,c->bc", VR, E,        optimize=opt_einsum), optimize=opt_einsum)))
  else:
    if B is None:
      errL = np.max(np.abs(np.einsum("ab,bc->ac", A.T.conj(), VL, optimize=opt_einsum) - np.einsum("bc,c->bc", VL, E.conj(), optimize=opt_einsum)))
      errR = np.max(np.abs(np.einsum("ab,bc->ac", A,          VR, optimize=opt_einsum) - np.einsum("bc,c->bc", VR, E,        optimize=opt_einsum)))
    else:
      errL = np.max(np.abs(np.einsum("ab,bc->ac", A.T.conj(), VL, optimize=opt_einsum) - np.einsum("ab,bc->ac", B.T.conj(), np.einsum("bc,c->bc", VL, E.conj(), optimize=opt_einsum), optimize=opt_einsum)))
      errR = np.max(np.abs(np.einsum("ab,bc->ac", A,          VR, optimize=opt_einsum) - np.einsum("ab,bc->ac", B,          np.einsum("bc,c->bc", VR, E,        optimize=opt_einsum), optimize=opt_einsum)))
  return np.maximum(errL, errR)

def re_ortho_normalize_ULUR(UL, UR, B, was_ortho = True, rotate_which = "left"):
  # U^{L,H} U^R should be a diagonal matrix
  # we want to have U^{L,H} U^R = I
  if was_ortho:
    if B is None:
      oU = np.einsum("am,am->m", UL.conj(), UR, optimize=True)
      print("oU = ",oU)
    else:
      oU = np.einsum("am,am->m", UL.conj(), np.einsum("ab,bm->am", B, UR, optimize=True), optimize=True)
    if rotate_which == "right":
      UR = np.einsum("am,m->am", UR, 1./oU)
    elif rotate_which == "left":
      UL = np.einsum("am,m->am", UL, 1./oU.conj())
  else:
    if B is None:
      oU = np.einsum("am,an->mn", UL.conj(), UR, optimize=True)
    else:
      oU = np.einsum("am,an->mn", UL.conj(), np.einsum("ab,bn->an", B, UR, optimize=True), optimize=True)
    if np.linalg.cond(oU) > 1./sys.float_info.epsilon:
      print("Warning!!! oU is singular")
    if rotate_which == "right":
      UR = np.einsum("am,mn->an", UR, sci.linalg.inv(oU))
    elif rotate_which == "left":
      UL = np.einsum("am,nm->an", UL, sci.linalg.inv(oU).conj())
  
  # normR / x = normL * x
  normR = np.linalg.norm(UR, axis=0)
  normL = np.linalg.norm(UL, axis=0)
  fac_eqnorm = np.sqrt(normR / normL)
  UR = np.einsum("am,m->am", UR, 1./fac_eqnorm)
  UL = np.einsum("am,m->am", UL, fac_eqnorm)
  normR = np.linalg.norm(UR, axis=0)
  normL = np.linalg.norm(UL, axis=0)
  #print("normR = \n",normR[0:10])
  #print("normL = \n",normL[0:10])
  return UL, UR

def rayleigh_ritz_method(A, B, VL, VR, type_e = "largest real", opt_einsum = True):
  if A is not None:
    Atilde = np.einsum("ba,bc->ac", VL.conj(), np.einsum("ab,bc->ac", A, VR, optimize=opt_einsum), optimize=opt_einsum)
  else:
    Atilde = np.einsum("ba,bc->ac", VL.conj(), VR, optimize=opt_einsum)
  if B is None:
    E, YL, YR = sci.linalg.eig(Atilde, left=True)
    YL, YR = re_ortho_normalize_ULUR(YL, YR, None)
    err_orth = check_orthonormal(B, YL, YR)
    print("err_orth = ",err_orth)
  else:
    Btilde = np.einsum("ba,bc->ac", VL.conj(), np.einsum("ab,bc->ac", B, VR, optimize=opt_einsum), optimize=opt_einsum)
    E, YL, YR = sci.linalg.eig(Atilde, b=Btilde, left=True)
    YL, YR = re_ortho_normalize_ULUR(YL, YR, Btilde)
  return sort_E_Y(type_e, E, YL, YR)

def proj_out(QL, QR, VL, VR, B, opt_einsum = True):
  if B is None:
    QL_ = QL - np.einsum("ab,bc->ac", VL, np.einsum("ba,bc->ac", VR.conj(), QL, optimize=opt_einsum), optimize=opt_einsum)
    QR_ = QR - np.einsum("ab,bc->ac", VR, np.einsum("ba,bc->ac", VL.conj(), QR, optimize=opt_einsum), optimize=opt_einsum)
  else:
    QL_ = QL - np.einsum("ab,bc->ac", VL, np.einsum("ba,bc->ac", VR.conj(), np.einsum("bd,dc->bc", B.T.conj(), QL, optimize=opt_einsum), optimize=opt_einsum), optimize=opt_einsum)
    QR_ = QR - np.einsum("ab,bc->ac", VR, np.einsum("ba,bc->ac", VL.conj(), np.einsum("bd,dc->bc", B,          QR, optimize=opt_einsum), optimize=opt_einsum), optimize=opt_einsum)
  return QL_, QR_

def extend_V(A, B, VL, VR, opt_einsum = True):
  VL_add = VL.copy()
  VR_add = VR.copy()
  if A is None:
    VL_add = np.einsum("ac,cb->ab", B.T.conj(), VL)
    VR_add = np.einsum("ac,cb->ab", B,          VR)
  else:
    VL_add = np.einsum("ac,cb->ab", A.T.conj(), VL)
    VR_add = np.einsum("ac,cb->ab", A,          VR)
    if B is not None:
      VL_B = np.einsum("ac,cb->ab", B.T.conj(), VL)
      VR_B = np.einsum("ac,cb->ab", B,          VR)
      VL_add = np.hstack((VL_add, VL_B))
      VR_add = np.hstack((VR_add, VR_B))
  VL_add, VR_add = proj_out(VL_add, VR_add, VL, VR, B)
  VL_add, VR_add = re_ortho_normalize_ULUR(VL_add, VR_add, B, False)
  return np.hstack((VL, VL_add)), np.hstack((VR, VR_add))

def solve_gevp_rgat(l, A, B, VL0, VR0, type_e = "largest real", type_err = "rel", thr = 1e-8, nter_max = 100, opt_einsum = True):
  # solve a few extreme eigenvalues and associated left/right eigenvectors
  # of generlized eigenvalue problem: A x = B x E
  # based on "Algorithm 1" in paper
  # "A method for computing a few eigenpairs of large generalized eigenvalue problems."
  # Appl. Numer. Math. 183, 108-117 (2023). by M. Alkilayh, L. Reichel, and Q. Ye
  print("\n======================================================================")
  print("solve a few eigenmodes of GEVP via iterative method")
  print("======================================================================")
  
  #======================================================================
  # Step 1: Initialize
  #======================================================================
  # initial checks
  n = VR0.shape[0]
  p = VR0.shape[1]
  if l > p:
    print("l should be <= p")
    exit(1)
  VL = VL0.copy()
  VR = VR0.copy()
  err_orth = check_orthonormal(B, VL, VR)
  print("err_orth = ",err_orth)
  if err_orth > thr:
    print("make VL and VR orthonornal")
    VL, VR = re_ortho_normalize_ULUR(VL, VR, B, False)
  
  # check if VL/VR are already solutions
  E0, YL, YR = rayleigh_ritz_method(A, B, VL, VR, type_e, opt_einsum)
  print("initial E:", E0)
  if A is not None:
    print("initial lifetimes:", units.ps/E0)
  else:
    print("initial lifetimes:", units.ps/((1./E0).real))
  VL = np.einsum("ab,bc->ac", VL, YL, optimize=opt_einsum)
  VR = np.einsum("ab,bc->ac", VR, YR, optimize=opt_einsum)
  err_orth = check_orthonormal(B, VL, VR)
  print("err_orth = ",err_orth)
  err_sol = check_solution(A, B, E0[0:l], VL[:,0:l], VR[:,0:l], opt_einsum)
  if err_sol < thr:
    print("1-step RR method is fine enough")
    return E0[0:l], VL[:,0:l], VR[:,0:l]
  
  t0 = timer()
  errE = 2 * thr
  err_sol = 2 * thr
  Eold = E0.copy()
  E = np.zeros_like(E0)
  
  for it in range(nter_max):
    #======================================================================
    # Step 1 or 4: Extend
    #======================================================================
    # extend VL/VR
    VLex, VRex = extend_V(A, B, VL, VR)
    err_orth = check_orthonormal(B, VLex, VRex)
    print("err_orth (step extend) = ",err_orth)
    
    #======================================================================
    # Step 2: Solve
    #======================================================================
    E, YL, YR = rayleigh_ritz_method(A, B, VLex[:,0:p], VRex[:,0:p], type_e, opt_einsum)
    print("Eex[0:p]:", E)
    E, YL, YR = rayleigh_ritz_method(A, B, VLex[:,p:2*p], VRex[:,p:2*p], type_e, opt_einsum)
    print("Eex[p:2p]:", E)
    Eex, YLex, YRex = rayleigh_ritz_method(A, B, VLex, VRex, type_e, opt_einsum)
    print("Eex:", Eex)
    VLex = np.einsum("ab,bc->ac", VLex, YLex, optimize=opt_einsum)
    VRex = np.einsum("ab,bc->ac", VRex, YRex, optimize=opt_einsum)
    err_orth = check_orthonormal(B, VLex, VRex)
    print("err_orth (step solve) = ",err_orth)
    
    #======================================================================
    # Step 3: Check
    #======================================================================
    E = Eex[0:p]
    print("E at iter. ",it,":", E)
    if A is not None:
      print("lifetimes:", units.ps/E)
    else:
      print("lifetimes:", units.ps/((1./E).real))
    VL = VLex[:,0:p].copy()
    VR = VRex[:,0:p].copy()
    del VLex
    del VRex
    err_sol = check_solution(A, B, E0[0:l], VL[:,0:l], VR[:,0:l], opt_einsum)  
    errE = compute_error(type_e, E[0:l], Eold[0:l], type_err, thr)
    if errE < thr and err_sol < thr:
      break
    else:
      Eold = E
  
  # final eigenvalues and (right) eigenvectors
  print("number of iterations = ",it," with err of E = ",errE," and err of EVP = ",err_sol)
  t1 = timer()
  print("time = ",t1-t0)
  return E0, E[0:l], VL[:,0:l], VR[:,0:l]