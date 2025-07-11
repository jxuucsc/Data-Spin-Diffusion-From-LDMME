#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
import units
from timeit import default_timer as timer

thr_deg = 1e-10
thr_print_mode_ana = 0.0001

##################################################
# some aux. functions
##################################################
def analyse_decay_mode(e, UR, UL, n_prt, rhopert, ob, Ekmn, Eref, w_dfde_rho, dir_ana = "analysis_ldmme/"):
  # L = V^R E V^{L,H} with E the eigenvalue matrix of L
  # by our construction, V^{L,H} V^R = I
  # (V^{R,T} o^T)_m (V^{L,H} rhopert)_m
  VR_ob = np.einsum("oa,am->mo", ob.conj(), UR, optimize=True)
  VLH_rhopert = np.einsum("mb,pb->mp", UL.conj().T, rhopert, optimize=True)
  ob_of_pert = np.einsum("pa,oa->po", rhopert, ob.conj(), optimize=True).real
  pert_and_ob = np.einsum("mo,mp->mpo", VR_ob, VLH_rhopert, optimize=True).real
  
  multU = np.transpose(UL.conj()*UR)
  abs_E2Eref = np.dot(np.abs(multU), np.abs(Ekmn-Eref))
  np.savetxt(dir_ana+"abs_E2Eref.dat", np.transpose([abs_E2Eref, e]))
  sigma_E2Eref = np.sqrt(np.dot(np.abs(multU), np.power(Ekmn-Eref, 2)))
  np.savetxt(dir_ana+"sigma_E2Eref.dat", np.transpose([sigma_E2Eref, e]))
  w_dfde_mode = np.dot(np.abs(multU), w_dfde_rho) / np.sum(w_dfde_rho)
  np.savetxt(dir_ana+"w_dfde_mode.dat", np.transpose([w_dfde_mode, e]))
  
  n_prt = min(UR.shape[1], n_prt)
  print("\nThe 4 observables and perturbations are:")
  print("n, sx, sy, sz")
  print("ob. proj. to VR = \n", np.where(np.abs(VR_ob[0:n_prt]) > thr_print_mode_ana, VR_ob[0:n_prt], 0))
  print("rhopert. proj. to VL = \n", np.where(np.abs(VLH_rhopert[0:n_prt]) > thr_print_mode_ana, VLH_rhopert[0:n_prt], 0))
  print("ob. of pert:\n", np.where(np.abs(ob_of_pert) > np.power(thr_print_mode_ana, 2), ob_of_pert, 0))
  for im in range(n_prt):
    print("\nmode ",im," with rate ",e[im])
    print("pert. and ob.:\n", np.where(np.abs(pert_and_ob[im]) > np.power(thr_print_mode_ana, 2), pert_and_ob[im], 0))
  return VR_ob, VLH_rhopert, pert_and_ob

def reorder_modes(e, UL, UR):
  tau = 1./np.real(e)
  ind = np.argsort(-np.abs(np.where(np.real(e) != 0, tau, 0)))
  tau = tau[ind]
  e = e[ind]
  UR = UR[:,ind]
  UL = UL[:,ind]
  return tau, e, UL, UR

##################################################
# some math functions
##################################################
# check Hermitian errors of a matrix
def check_Hermitian(M,s):
  if M.shape[-1] != M.shape[-2]:
    return
  print("\nCheck Hermitian errors of matrix "+s)
  err = np.abs(M - 0.5*(M + M.swapaxes(0,1).conj()))
  print("max Hermitian error: ",np.amax(err))
  print("indices for max Hermitian error: ",np.unravel_index(np.argmax(err, axis=None), err.shape))
  print("")

def check_mask_rho(mask_rho, Ek, emin, emax, mask_Ek):
  nk = Ek.shape[0]
  nb = Ek.shape[1]
  for ik in range(nk):
    for b1 in range(nb):
      in_range_1 = Ek[ik,b1] > emin and Ek[ik,b1] < emax
      if in_range_1 and not mask_Ek[ik,b1]:
        print("mask_Ek is not right")
        exit(1)
      if not in_range_1 and mask_Ek[ik,b1]:
        print("mask_Ek is not right")
        exit(1)
      for b2 in range(nb):
        in_range_2 = Ek[ik,b2] > emin and Ek[ik,b2] < emax
        if (in_range_1 and in_range_2) and not mask_rho[ik,b1,b2]:
          print("mask_rho is not right")
          exit(1)
        if (not in_range_1 or not in_range_2) and mask_rho[ik,b1,b2]:
          print("mask_rho is not right")
          exit(1)
  print("mask_Ek and mask_rho are right")


def wrap(k, c=np.zeros(3)):
  r = k - c - np.floor(k - c + 0.5)
  r = np.where(abs(r-0.5)<1e-6, -0.5, r)
  return r+c

def normalize_vector(v, axis_=1):
  if axis_ == 1:
    n = np.linalg.norm(v, axis=1)
    return np.einsum("ab,a->ab", v, 1./n, optimize=True)
  elif axis_ == 0:
    n = np.linalg.norm(v, axis=0)
    return np.einsum("ab,b->ab", v, 1./n, optimize=True)
  else:
    print("axis_ = ",axis_," is not allowed")
    exit(1)

def biorthogonalize(A, B, trans=False):
  # let A and B biorthogonal with A unchanged
  if trans:
    ncomm = min(A.shape[0], B.shape[0])
    B_tmp = np.einsum("pa,p->pa", A[0:ncomm], np.einsum("pa,pa->p", A[0:ncomm].conj(), B[0:ncomm], optimize=True), optimize=True)
    B = B - np.einsum("oa,po->pa", A, np.einsum("oa,pa->po", A.conj(), B, optimize=True), optimize=True)
    B[0:ncomm] = B[0:ncomm] + B_tmp
  else:
    ncomm = min(A.shape[1], B.shape[1])
    B_tmp = np.einsum("am,m->am", A[:,0:ncomm], np.einsum("am,am->m", A[:,0:ncomm].conj(), B[:,0:ncomm], optimize=True), optimize=True)
    B = B - np.einsum("am,mn->am", A, np.einsum("am,an->mn", A.conj(), B, optimize=True), optimize=True)
    B[:,0:ncomm] = B[:,0:ncomm] + B_tmp
  return B

def make_U_hermitian(U, nk, nb, mask_rho):
  U_full = np.zeros((nk*nb*nb,U.shape[1]), np.complex128)
  U_full[mask_rho,:] = U
  U_full = U_full.reshape(nk,nb,nb,U.shape[1])
  U_full = 0.5 * (U_full + np.transpose(U_full.conj(), (0,2,1,3)))
  U = U_full.reshape(nk*nb*nb,U.shape[1])[mask_rho,:]
  return U

def re_ortho_normalize_ULUR(UL, UR, was_ortho = True, rotate_which = "left"):
  # U^{L,H} U^R should be a diagonal matrix
  # we want to have U^{L,H} U^R = I
  if was_ortho:
    oU = np.einsum("am,am->m", UL.conj(), UR, optimize=True)
    if rotate_which == "right":
      UR = np.einsum("am,m->am", UR, 1./oU)
    elif rotate_which == "left":
      UL = np.einsum("am,m->am", UL, 1./oU.conj())
  else:
    oU = np.einsum("am,an->mn", UL.conj(), UR, optimize=True)
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

def get_U_rotate_within_deg(UL, UR, ob_scale, m0, m1):
  S_UR = np.einsum("oa,am->om", ob_scale[m0:m1].conj(), UR[:,m0:m1])
  print("<s|UR>:\n", S_UR)
  UR_rotate = np.einsum("am,mn->an", UR[:,m0:m1], sci.linalg.inv(S_UR))
  UL_rotate = np.einsum("am,nm->an", UL[:,m0:m1], S_UR.conj())
  normR_rotate = np.linalg.norm(UR_rotate, axis=0)
  normL_rotate = np.linalg.norm(UL_rotate, axis=0)
  fac_eqnorm_rotate = np.sqrt(normR_rotate / normL_rotate)
  UR_rotate = np.einsum("am,m->am", UR_rotate, 1./fac_eqnorm_rotate)
  UL_rotate = np.einsum("am,m->am", UL_rotate, fac_eqnorm_rotate)
  normR_rotate = np.linalg.norm(UR_rotate, axis=0)
  normL_rotate = np.linalg.norm(UL_rotate, axis=0)
  print("normR_rotate = ", normR_rotate)
  print("normL_rotate = ", normL_rotate)
  S_UR_rotate = np.einsum("oa,am->om", normalize_vector(ob_scale[m0:m1]).conj(), UR_rotate)
  print("<s|UR>:\n", S_UR_rotate)
  oUs_rotate = np.einsum("ba,bc->ac", UL_rotate.conj(), UR_rotate)
  print("oUs_rotate:\n",oUs_rotate)
  #print("UL^H -L UR:\n", np.einsum("am,ab,bn->mn", UL[:,m0:m1].conj(), -L, UR[:,m0:m1], optimize=True))
  #print("UL^H -L UR (rotated):\n", np.einsum("am,ab,bn->mn", UL_rotate.conj(), -L, UR_rotate, optimize=True))
  return UL_rotate,UR_rotate

def compute_Ok_U(ob, UL, UR, mask_rho, nk, nb):
  nm = UR.shape[1]
  ob_full = np.zeros((ob.shape[0],nk*nb*nb),np.complex128)
  ob_full[:,mask_rho] = ob
  ob_full = ob_full.reshape(ob.shape[0],nk,nb,nb)
  UL_full = np.zeros((nk*nb*nb,nm),np.complex128)
  UL_full[mask_rho,:] = UL
  UL_full = UL_full.reshape(nk,nb,nb,nm)
  UR_full = np.zeros((nk*nb*nb,nm),np.complex128)
  UR_full[mask_rho,:] = UR
  UR_full = UR_full.reshape(nk,nb,nb,nm)
  Ok_UL = np.einsum("okab,kbam->mokb", ob_full, UL_full)
  Ok_UR = np.einsum("okab,kbam->mokb", ob_full, UR_full)
  return Ok_UL, Ok_UR

# exp. cos fit of lifetimes and diffusion lengths
def decay_cosine(t, beta, omega, phi):
  s = np.exp(-beta*t) * np.cos(omega*t + phi)
  return s
def decay_cosine_phi0(t, beta, omega):
  s = np.exp(-beta*t) * np.cos(omega*t)
  return s
def residuals(args, t, s):
  return s - decay_cosine(t, *args)
def residuals2(args, t, s):
  return s - decay_cosine_phi0(t, *args)
def residuals3(beta, t, s):
  return s - np.exp(-beta*t)
def fit_Ot_expcos(t, Ot, comment, tauinv=0.001, w=0.1/2.3505175675871e5/units.ps):
  phi = 0 # phase
  params = tauinv, w, phi
  params2 = tauinv, w
  params_lsq,  _ = sci.optimize.leastsq(residuals3, tauinv,  args=(t*units.ps, Ot/Ot[0]), maxfev=9999999)
  params_lsq2, _ = sci.optimize.leastsq(residuals2, params2, args=(t*units.ps, Ot/Ot[0]), maxfev=9999999)
  params_lsq3, _ = sci.optimize.leastsq(residuals,  params,  args=(t*units.ps, Ot/Ot[0]), maxfev=9999999)
  print(comment+" (exp): tau = ",1/params_lsq[0]," ps")
  print(comment+" (exp cos phi=0): tau = ",1/params_lsq2[0]," ps"," period = ",2*np.pi/params_lsq2[1],"ps")
  print(comment+" (exp cos): tau = ",1/params_lsq3[0]," ps"," period = ",2*np.pi/params_lsq3[1],"ps"," phi = ",params_lsq3[2])
def fit_Ox_expcos(x, Ox, comment, ldiffinv=0.001, wl=0.1/2.3505175675871e5/(1e9*units.meter)):
  phi = 0 # phase
  params = ldiffinv, wl, phi
  params2 = ldiffinv, wl
  params_lsq,  _ = sci.optimize.leastsq(residuals3, ldiffinv, args=(x*(1e9*units.meter), Ox/Ox[0]), maxfev=9999999)
  params_lsq2, _ = sci.optimize.leastsq(residuals2, params2,  args=(x*(1e9*units.meter), Ox/Ox[0]), maxfev=9999999)
  params_lsq3, _ = sci.optimize.leastsq(residuals,  params,   args=(x*(1e9*units.meter), Ox/Ox[0]), maxfev=9999999)
  print(comment+" (exp): tau = ",1e-3/params_lsq[0]," micro-meter")
  print(comment+" (exp cos phi=0): tau = ",1e-3/params_lsq2[0]," micro-meter"," period = ",2*np.pi*1e-3/params_lsq2[1])
  print(comment+" (exp cos): tau = ",1e-3/params_lsq3[0]," micro-meter"," period = ",2*np.pi*1e-3/params_lsq3[1],"micro-meter"," phi = ",params_lsq3[2])