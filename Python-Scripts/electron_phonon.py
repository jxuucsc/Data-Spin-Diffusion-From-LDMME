#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
sys.path.insert(1, '/home/xujq/codes/jdftx/tools/solve_linearDMME/')
from lattice import *
from electron import *
import ldbd_files as ldbd
from help_solve_tl_fromldmme_ldbdfiles import *
import units
from timeit import default_timer as timer

thr_deg = 1e-10
thr_nex = 1

##################################################
# carrier transport from linearized DMME
##################################################
def compute_carrier_transport_ldmme(rhoE1, elec, latt):
  print("\n--------------------------------------------------")
  print("carrier transport from linearized DMME")
  print("--------------------------------------------------")
  cond = -np.real(np.dot(rhoE1, elec.vx.conj())) / (elec.nk_full * latt.cell_size)
  if latt.dim == 3:
    print("conductivity in S/m = ",cond/units.Ohm_meter)
  elif latt.dim == 2:
    print("conductivity in S = ",cond/units.Ohm)
  if not elec.assumeMetal:
    mob = cond / elec.n_carr
    print("mobility in cm^2/V/s = ",mob*units.cm2byVs)
  elif elec.nv > 0 and elec.nv < elec.nb:
    if np.abs(elec.n_carr) > 1e-40:
      mob = cond / elec.n_carr
      print("mobility (n=|ne-nh|) in cm^2/V/s = ",mob*units.cm2byVs)
    mob = cond / (elec.ne + elec.nh)
    print("mobility (n=ne+nh) in cm^2/V/s = ",mob*units.cm2byVs)

def compute_rhoeqE1(mLpinv, DfDk):
  return np.einsum("ab,b->a", mLpinv, DfDk, optimize=True)

class electron_phonon(object):
  
  def init(self, latt, elec, scale_sc, typeP, trunc_outer, Lsc_from, write_Lsc):
    print("\n##################################################")
    print("read and compute e-ph quantities")
    print("##################################################")
    t0 = timer()
    self.latt = latt
    self.elec = elec
    self.nkp = elec.nkp
    self.nk = elec.nk
    self.nb = elec.nb
    
    self.kp = ldbd.read_ldbd_kpair(self.nkp, elec.nb, elec.nv)
    os.system("mkdir restart_ldmme")
    self.Lsc = np.array([])
    if Lsc_from == "P":
      self.P = scale_sc * ldbd.read_ldbd_P(self.nkp, elec.nb, elec.nk, self.kp, typeP, elec.nv)
      if not os.path.exists("restart_ldmme/Lsc.bin"):
        self.Lsc = self.compute_L_from_P(self.P, elec.fk_full)
    elif Lsc_from == "dmd":
      self.Lsc = (scale_sc * 0.5 * 2 * np.pi / elec.nk_full) * ldbd.read_ldbd_Lsc(self.nkp, self.nb, self.nk, self.kp)
      self.Lsc = self.Lsc[:,self.elec.mask_rho][self.elec.mask_rho,:]
    elif Lsc_from == "dmd trunc":
      self.Lsc = (scale_sc * 0.5 * 2 * np.pi / elec.nk_full) * ldbd.read_ldbd_Lsc_trunc(self.nkp, self.nb, self.nk, self.kp, self.elec.size_rho_in, self.elec.ind_k_rho, self.elec.nb_k)
    
    if Lsc_from == "P":
      self.reshape_and_trunc_outer_scatt(trunc_outer)
    #self.Lsc[:,self.elec.mask_rho_outer] = 0
    #self.Lsc[self.elec.mask_rho_outer,:] = 0
    
    if os.path.exists("restart_ldmme/Lsc.bin"):
      self.Lsc = np.fromfile("restart_ldmme/Lsc.bin", np.complex128).reshape(self.elec.size_rho_in, self.elec.size_rho_in)
    elif write_Lsc:
      self.Lsc.tofile("restart_ldmme/Lsc.bin")
    
    if not elec.has_Bext:
      self.L = self.Lsc + np.diag(elec.dE)
    else:
      self.L = self.Lsc + elec.Lcoh
    t1 = timer()
    print("time for e-ph setup: ",t1-t0)
    
    #-------------------------------------------------
    # carrier relaxation and transport
    #-------------------------------------------------
    self.elec.dfdek = self.elec.fk * (self.elec.fk - 1) / self.elec.T
    if Lsc_from == "P":
      self.compute_carrier_transport_conventional(self.elec.fk, self.elec.dfdek) # Boltzmann
      del self.P
    
    # analysis diagonal elements of Lsc
    Gamma_kmn = -np.diag(self.Lsc)
    Gamma_kn = Gamma_kmn[self.elec.mask_bdiag]
    np.savetxt("analysis_ldmme/Gammakn.dat", np.transpose([self.elec.Ek, np.real(Gamma_kn), np.imag(Gamma_kn)]))
    np.savetxt("analysis_ldmme/Gammakmn.dat", np.transpose([self.elec.Ekmn_mat, np.real(Gamma_kmn), np.imag(Gamma_kmn)]))
    Gamma_avg = np.dot(Gamma_kn, self.elec.dfdek) / np.sum(self.elec.dfdek)
    print("carrier lifetime (rate average) in ps: ", 2.4188843265857e-5/Gamma_avg.real)
    Gamma_kmn_avg = np.dot(Gamma_kmn, self.elec.dfde_mat) / np.sum(self.elec.dfde_mat)
    print("effetive lifetime (single element of rho, rate average) in ps: ", 2.4188843265857e-5/Gamma_kmn_avg.real)
    t2 = timer()
    print("time for e-ph: ",t2-t1)
  
  ##################################################
  # master equation related
  ##################################################
  def compute_Lii_from_P(self, ik, P, fk):
    Lii = -np.einsum("ac,bd->abcd", np.eye(self.nb), np.einsum("qdbee,qe->bd", P[ik], fk, optimize=True), optimize=True) \
          -np.einsum("ac,bd->abcd", np.einsum("qe,qeeac->ac", 1-fk, P[:,ik].conj(), optimize=True), np.eye(self.nb), optimize=True)
    return (Lii + np.transpose(Lii, (1,0,3,2)).conj())
  def compute_Lij_from_P(self, P, fk):
    Lij = np.einsum("ka,kqabcd->kabqcd", 1-fk, P, optimize=True) + \
          np.einsum("qkcdab,kb->kabqcd", P.conj(), fk, optimize=True)
    return (Lij + np.transpose(Lij, (0,2,1,3,5,4)).conj())
  def compute_L_from_P(self, P, fk):
    L = self.compute_Lij_from_P(P, fk).reshape(self.nk,self.nb*self.nb,self.nk,self.nb*self.nb)
    for ik in range(self.elec.nk):
      L[ik,:,ik] = L[ik,:,ik] + self.compute_Lii_from_P(ik, P, fk).reshape(self.nb*self.nb,self.nb*self.nb)
    return (1./2/self.elec.nk_full)*L.reshape(self.nk*self.nb*self.nb,self.nk*self.nb*self.nb)
  
  ##################################################
  # for ruling out states outside [emin, emax]
  ##################################################
  def reshape_and_trunc_outer_scatt(self, trunc_outer):
    self.P = np.transpose(self.P.reshape(self.nk,self.nk,self.nb*self.nb,self.nb*self.nb), (0,2,1,3)).reshape(self.nk*self.nb*self.nb,self.nk*self.nb*self.nb)
    if trunc_outer and np.count_nonzero(self.elec.mask_Ek) < self.nk*self.nb:
      self.P = self.P[:,self.elec.mask_rho][self.elec.mask_rho,:]
      if not os.path.exists("restart_ldmme/Lsc.bin"):
        self.Lsc = self.Lsc[:,self.elec.mask_rho][self.elec.mask_rho,:]
  
  ##################################################
  # conventional carrier transport
  ##################################################
  def compute_carrier_transport_conventional(self, fk, dfdek):
    print("\n--------------------------------------------------")
    print("carrier transport from conventional method")
    print("--------------------------------------------------")
    t0 = timer()
    Pdd = np.real(self.P[:,self.elec.mask_bdiag][self.elec.mask_bdiag,:])
    vkn = np.real(self.elec.vmat[:,self.elec.mask_bdiag])
    nstate = fk.shape[0]
    
    vnorm = np.linalg.norm(vkn, axis=0)
    vnorm_inv = np.where(vnorm > 1e-15, 1./vnorm, 0)
    vkn_norm = np.einsum("di,i->di", vkn, vnorm_inv)
    vij_inorm = np.einsum("di,dj->ij", vkn_norm, vkn)
    vij_ijnorm = np.einsum("ij,j->ij", vij_inorm, vnorm_inv)
    
    tau_inv_ij = np.einsum("ij,j->ij", Pdd, fk) + np.einsum("j,ji->ij", 1-fk, Pdd)
    np.fill_diagonal(tau_inv_ij, 0)
    tau_inv = np.sum(tau_inv_ij, axis=1) / self.elec.nk_full
    taum_inv = np.einsum("ij,ij->i", tau_inv_ij, vij_ijnorm) / self.elec.nk_full
    taum_inv = tau_inv - taum_inv
    taum2_inv = np.einsum("ij,ij,i->i", tau_inv_ij, vij_inorm, vnorm_inv) / self.elec.nk_full
    taum2_inv = tau_inv - taum2_inv
    
    tau = np.where(tau_inv > 1e-20, 1./tau_inv, 0)
    taum = np.where(taum_inv > 1e-20, 1./taum_inv, 0)
    taum2 = np.where(taum2_inv > 1e-20, 1./taum2_inv, 0)
    tau_inv_avg = np.dot(tau_inv, dfdek) / np.sum(dfdek)
    tau_avg = np.dot(tau, dfdek) / np.sum(dfdek)
    taum_inv_avg = np.dot(taum_inv, dfdek) / np.sum(dfdek)
    taum_avg = np.dot(taum, dfdek) / np.sum(dfdek)
    taum2_inv_avg = np.dot(taum2_inv, dfdek) / np.sum(dfdek)
    taum2_avg = np.dot(taum2, dfdek) / np.sum(dfdek)
    print("carrier lifetime (rate average) in ps: ", units.ps/tau_inv_avg)
    print("carrier lifetime (time average) in ps: ", units.ps*tau_avg)
    print("momentum lifetime (rate average) in ps: ", units.ps/taum_inv_avg)
    print("momentum lifetime (time average) in ps: ", units.ps*taum_avg)
    print("momentum lifetime (type 2, rate average) in ps: ", units.ps/taum2_inv_avg)
    print("momentum lifetime (type 2, time average) in ps: ", units.ps*taum2_avg)
    
    #v2_avg = np.zeros(3, np.float64)
    #for i in range(3):
    #  v2_avg[i] = np.dot(v2_kn[i], dfdek) / np.sum(dfdek)
    v2_avg = np.dot(np.power(vkn, 2), dfdek) / np.sum(dfdek)
    print("average of v^2 = ",v2_avg)
    print("Fermi velocity in m/s = ",np.sqrt(v2_avg)*units.mbys)
    
    cond_tau = np.dot(np.power(vkn,2), -dfdek*tau) / (self.elec.nk_full * self.latt.cell_size)
    cond_taum = np.dot(np.power(vkn,2), -dfdek*taum) / (self.elec.nk_full * self.latt.cell_size)
    cond_taum2 = np.dot(np.power(vkn,2), -dfdek*taum2) / (self.elec.nk_full * self.latt.cell_size)
    if self.latt.dim == 3:
      print("conductivity using tau in S/m = ",cond_tau/units.Ohm_meter)
      print("conductivity using taum in S/m = ",cond_taum/units.Ohm_meter)
      print("conductivity using taum2 in S/m = ",cond_taum2/units.Ohm_meter)
    elif self.latt.dim == 2:
      print("conductivity using tau in S = ",cond_tau/units.Ohm)
      print("conductivity using taum in S = ",cond_taum/units.Ohm)
      print("conductivity using taum2 in S = ",cond_taum2/units.Ohm)
    if not self.elec.assumeMetal:
      mob_tau = cond_tau / self.elec.n_carr
      print("mobility using tau in cm^2/V/s = ",mob_tau*units.cm2byVs)
      mob_taum = cond_taum / self.elec.n_carr
      print("mobility using taum in cm^2/V/s = ",mob_taum*units.cm2byVs)
      mob_taum2 = cond_taum2 / self.elec.n_carr
      print("mobility using taum2 in cm^2/V/s = ",mob_taum2*units.cm2byVs)
    elif self.elec.nv > 0 and self.elec.nv < self.nb:
      if np.abs(self.elec.n_carr) > 1e-40:
        mob_tau = cond_tau / self.elec.n_carr
        print("mobility (n=|ne-nh|) using tau in cm^2/V/s = ",mob_tau*units.cm2byVs)
        mob_taum = cond_taum / self.elec.n_carr
        print("mobility (n=|ne-nh|) using taum in cm^2/V/s = ",mob_taum*units.cm2byVs)
        mob_taum2 = cond_taum2 / self.elec.n_carr
        print("mobility (n=|ne-nh|) using taum2 in cm^2/V/s = ",mob_taum2*units.cm2byVs)
      mob_tau = cond_tau / (self.elec.ne + self.elec.nh)
      print("mobility (n=ne+nh) using tau in cm^2/V/s = ",mob_tau*units.cm2byVs)
      mob_taum = cond_taum / (self.elec.ne + self.elec.nh)
      print("mobility (n=ne+nh) using taum in cm^2/V/s = ",mob_taum*units.cm2byVs)
      mob_taum2 = cond_taum2 / (self.elec.ne + self.elec.nh)
      print("mobility (n=ne+nh) using taum2 in cm^2/V/s = ",mob_taum2*units.cm2byVs)
    t1 = timer()
    print("time for compute_carrier_transport_conventional: ",t1-t0)