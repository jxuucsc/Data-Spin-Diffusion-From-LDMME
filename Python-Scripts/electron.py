#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
sys.path.insert(1, '/home/xujq/codes/jdftx/tools/solve_linearDMME/')
from lattice import *
import ldbd_files as ldbd
from help_solve_tl_fromldmme_ldbdfiles import *
import units
from kmapper import *
from timeit import default_timer as timer

thr_deg = 1e-10
thr_nex = 1

def fermi(Ek, T, mu):
  return 1./(np.exp((Ek - mu) / T) + 1);

class electron(object):
  
  def init(self, latt, excess_density, mu, scale_coh, Bext, renorm_spinpert, trunc_outer, dirv): #, if_reset_erange, nkbt):
    print("\n##################################################")
    print("read and compute electronic quantities")
    print("##################################################")
    t0 = timer()
    self.latt = latt
    
    self.assumeMetal, self.has_h, self.nb, self.nv, self.nk_full, self.nk, self.kmesh, self.nkp, self.T, self.emin, self.emax = ldbd.read_ldbd_size()
    self.k = ldbd.read_ldbd_kvec(self.nk)
    self.kmap = kmapper()
    #self.kmap.init(self.kmesh, self.k)
    latt.get_valley_k(self.k)
    
    self.Ek_full = ldbd.read_ldbd_ek(self.nk,self.nb)
    self.Ek = self.Ek_full.copy()
    print("Ek:",self.Ek[0:min(self.nk,10)])
    if self.nv > 0:
      self.vbm = np.max(self.Ek[:,0:self.nv])
      print("vbm in eV = ",self.vbm*units.eV)
    if self.nv < self.nb:
      self.cbm = np.min(self.Ek[:,self.nv:self.nb])
      print("cbm in eV = ",self.cbm*units.eV)
    if self.nv > 0 and self.nv < self.nb:
      print("gap in eV = ",(self.cbm-self.vbm)*units.eV)
    self.Ekmn_mat = 0.5 * (self.Ek[:,:,None] + self.Ek[:,None,:])
    self.dE = scale_coh * ((-1j*(self.Ek[:,:,None] - self.Ek[:,None,:])).reshape(-1))
    
    if excess_density is None or (excess_density == 0 and not self.assumeMetal) or self.assumeMetal:
      self.mu = mu
    else:
      self.excess_density = excess_density * np.power(units.cm, latt.dim) # input excess density unit is cm^-d
      self.mu = self.find_mu(self.excess_density)
    self.fk = fermi(self.Ek, self.T, self.mu)
    self.fk_full = self.fk.copy()
    print("fk:",self.fk[0:min(self.nk,10)])
    self.dfdek = self.fk * (self.fk - 1) / self.T
    self.dfde_mat = self.compute_dfde_mat()
    self.n_carr, self.ne, self.nh = self.compute_carrier_density(self.fk)
    #if trunc_outer and if_reset_erange:
    #  self.reset_erange(nkbt)
    
    self.sop = ldbd.read_ldbd_smat(self.nk,self.nb)
    print("sk:\n", self.sop[0:min(self.nk,3)])
    self.has_Bext = np.linalg.norm(Bext) > 1e-20
    if self.has_Bext:
      self.Hcoh = np.einsum("ka,ab->kab", self.Ek, np.eye(self.nb, dtype=np.float64)) + np.einsum("i,kiab->kab", Bext, self.sop)
    self.vmat_full = np.transpose(ldbd.read_ldbd_vmat(self.nk,self.nb), (1,0,2,3))
    self.vmat = self.vmat_full.copy()
    self.dirv = dirv
    self.vx = self.vmat[dirv]
    self.DfDk = self.compute_DfDk(self.fk)
    self.ob_all,self.rhopert_all = self.set_rhopert_and_ob_all(renorm_spinpert)
    del self.sop
    
    #--------------------------------------------------
    # for ruling out states outside [emin, emax]
    #--------------------------------------------------
    if os.path.exists("brange_inner/bStart_k.bin") and os.path.exists("brange_inner/bEnd_k.bin"):
      self.set_mask_Ek_from_brange_inner()
    else:
      self.mask_Ek = np.logical_and(self.Ek > self.emin, self.Ek < self.emax)
    print("mask_Ek:\n", self.mask_Ek)
    self.mask_rho = np.logical_and(self.mask_Ek[:,None], self.mask_Ek[:,:,None])
    print("\nnumber of states within [emin,emax]: ",np.count_nonzero(self.mask_Ek)," (",self.nk*self.nb,")")
    print("number of elements of rho left: ",np.count_nonzero(self.mask_rho)," (",self.nk*self.nb*self.nb,")")
    self.check_mask_rho()
    if not trunc_outer:
      self.mask_Ek.fill(True)
      self.mask_rho.fill(True)
    self.size_rho_in = np.count_nonzero(self.mask_rho)
    self.nmode = self.size_rho_in
    self.ind_k_state, self.ind_k_rho, self.nb_k = self.set_ind_state_rho(self.mask_Ek, self.mask_rho)
    
    self.reshape_and_trunc_outer_elec(trunc_outer)
    
    if self.has_Bext:
      self.Lcoh = self.compute_Lcoh_from_Hcoh()
    self.Lv = self.compute_Lv_from_vx()
    
    t1 = timer()
    print("time for electron: ",t1-t0)

  
  ##################################################
  # Fermi-Dirac related
  ##################################################
  def find_mu(self, target_nex):
    mul = self.emin
    if self.nv == 0:
      mul = mul - 50*self.T
    muh = self.emax
    if self.nv == self.nb:
      muh = muh + 50*self.T
    nex, mu, mul, muh = self.mu_range(target_nex, mul, muh)
    niter = 0
    err = abs(nex - target_nex)/np.power(units.cm, self.latt.dim)
    while err > thr_nex:
      nex, mu, mul, muh = self.mu_range(target_nex, mul, muh)
      niter = niter + 1
      err = abs(nex - target_nex)/np.power(units.cm, self.latt.dim)
      if niter > 1e5:
        break
    print("number of iterations: ",niter," , err = ",err)
    print("final mu = %13.6e eV" % (mu*units.eV))
    return mu
  
  def mu_range(self, target_nex, mul, muh):
    mu_mid = 0.5 * (mul + muh)
    n,ne,nh = self.compute_carrier_density(fermi(self.Ek, self.T, mu_mid), False)
    nex = ne - nh
    if nex > target_nex:
      return nex, mu_mid, mul, mu_mid
    else:
      return nex, mu_mid, mu_mid, muh
  
  def compute_carrier_density(self, fk, print_info=True):
    ne = np.sum(fk[:,self.nv:]) / (self.nk_full * self.latt.cell_size)
    if print_info:
      print("\nelectron density = %10.3e in cm^-d" % (ne/np.power(units.cm,self.latt.dim)))
    nh = np.sum(1-fk[:,0:self.nv]) / (self.nk_full * self.latt.cell_size)
    if print_info:
      print("hole density = %10.3e in cm^-d" % (nh/np.power(units.cm,self.latt.dim)))
    if print_info:
      print("excess density = %10.3e in cm^-d\n" % ((ne-nh)/np.power(units.cm,self.latt.dim)))
    n = 0
    if not self.assumeMetal:
      if not self.has_h:
        n = ne
      else:
        n = nh
    elif self.nv > 0 and self.nv < self.nb:
      n = ne - nh
    return n, ne, nh
  
  def compute_dfde_mat(self):
    r = np.zeros((self.nk,self.nb,self.nb), np.float64)
    for ik in range(self.nk):
      for b1 in range(self.nb):
        for b2 in range(self.nb):
          if np.abs(self.Ek[ik,b1] - self.Ek[ik,b2]) < thr_deg:
            favg = 0.5*(self.fk[ik,b1] + self.fk[ik,b2])
            r[ik,b1,b2] = favg * (favg - 1) / self.T
          else:
            r[ik,b1,b2] = (self.fk[ik,b1] - self.fk[ik,b2]) / (self.Ek[ik,b1] - self.Ek[ik,b2])
    return r.reshape(-1)
  
  def compute_DfDk(self, fk):
    DfDk = np.zeros_like(self.vx)
    for ik in range(self.nk):
      for b1 in range(self.nb):
        for b2 in range(self.nb):
          dfde = 0
          if b1 == b2:
            dfde = fk[ik,b1] * (fk[ik,b1]- 1) / self.T
          elif np.abs(self.Ek[ik,b1] - self.Ek[ik,b2]) > thr_deg:
            dfde = (fk[ik,b1] - fk[ik,b2]) / (self.Ek[ik,b1] - self.Ek[ik,b2])
          DfDk[ik,b1,b2] = dfde * self.vx[ik,b1,b2]
    return DfDk
  
  '''
  def reset_erange(self, nkbt):
  	print("\nreset energy range:")
  	emin_old = self.emin
  	emax_old = self.emax
  	if not self.assumeMetal:
  		if not self.has_h:
  			self.emax = np.maximum(self.mu, self.cbm) + self.T * nkbt
  		else:
  			self.emin = np.minimum(self.mu, self.vbm) - self.T * nkbt
  	else:
  		self.emin = self.mu - self.T * nkbt
  		self.emax = self.mu + self.T * nkbt
  	print("emin = %.6f (%.6f)  emax = %.6f (%.6f)" % (self.emin, emin_old, self.emax, emax_old))
  '''
  
  ##################################################
  # setup pert. density-matrix and observable operator
  ##################################################
  def set_rhopert_and_ob_all(self, renorm_spinpert):
    # setup perturbation density matrices and observable operator matrices
    nvp = self.latt.valpol_k.shape[0]
    self.valpol_mat = np.einsum("ik,a->ika", self.latt.valpol_k, np.ones(self.nb*self.nb, np.int32), optimize=True).reshape(nvp,self.nk,self.nb,self.nb)
    nm = 7 * (1 + nvp)
    
    ob = np.zeros((nm,self.nk,self.nb,self.nb),np.complex128)
    for b in range(self.nb):
      ob[0,:,b,b] = 1
    for idir in range(3):
      ob[idir+1] = self.sop[:,idir]
      ob[idir+4] = 0.5 * (np.einsum("kab,kbc->kac", self.sop[:,idir], self.vx) + np.einsum("kab,kbc->kac", self.vx, self.sop[:,idir]))
    for ivp in range(nvp):
      for i in range(7):
        ob[ivp*7+7+i] = ob[i] * self.valpol_mat[ivp]
    ob = ob.reshape(-1,self.nk*self.nb*self.nb)
    
    rhopert = ob * self.dfde_mat
    if renorm_spinpert:
      rhopert = rhopert.reshape(nm,self.nk,self.nb*self.nb)
      npert_k = np.sqrt(np.real(np.einsum("ka,ka->k", rhopert[0], rhopert[0])))
      spert_k = np.sqrt(np.real(np.einsum("ika,ika->k", rhopert[1:4].conj(), rhopert[1:4])))
      fac_renorm = np.where(spert_k > 1e-30, npert_k / spert_k, 1e30*npert_k)
      rhopert[1:4] = np.einsum("k,ika->ika", fac_renorm, rhopert[1:4])
      rhopert = rhopert.reshape(-1,self.nk*self.nb*self.nb)
    rhopert = normalize_vector(rhopert)
    return ob, rhopert
  
  def select_reset_rhopert_and_ob(self, ind_pert_sel, ind_ob_sel):
    ob = self.ob_all[ind_ob_sel]
    fac_scale_ob = np.linalg.norm(self.ob_all[0])
    ob_scale = (1/fac_scale_ob) * ob
    rhopert = self.rhopert_all[ind_pert_sel]
    return ob, ob_scale, rhopert
  
  ##################################################
  # for master equation
  ##################################################
  def compute_Lcoh_from_Hcoh(self):
    L = np.zeros((self.size_rho_in,self.size_rho_in), np.complex128)
    mask_rho_tmp = self.mask_rho.reshape(self.nk,self.nb*self.nb)
    for ik in range(self.nk):
      Ltmp = np.zeros((self.nb,self.nb,self.nb,self.nb), np.complex128)
      for b in range(self.nb):
        Ltmp[:,b,:,b] = (-1j) * self.Hcoh[ik]
      for a in range(self.nb):
        Ltmp[a,:,a,:] = Ltmp[a,:,a,:] + 1j * self.Hcoh[ik].T
      Ltmp2 = Ltmp.reshape(self.nb*self.nb, self.nb*self.nb)[:,mask_rho_tmp[ik]][mask_rho_tmp[ik],:]
      L[self.ind_k_rho[ik]:self.ind_k_rho[ik+1], self.ind_k_rho[ik]:self.ind_k_rho[ik+1]] = Ltmp2
    return L
    '''
    L = np.zeros((self.nk,self.nb,self.nb,self.nk,self.nb,self.nb), np.complex128)
    for ik in range(self.nk):
      for b in range(self.nb):
        L[ik,:,b,ik,:,b] = (-1j) * self.Hcoh[ik]
      for a in range(self.nb):
        L[ik,a,:,ik,a,:] = L[ik,a,:,ik,a,:] + 1j * self.Hcoh[ik].T
    return L.reshape(self.nk*self.nb*self.nb,self.nk*self.nb*self.nb)
    '''
  
  def compute_Lv_from_vx(self):
    #'''
    Lv = np.zeros((self.size_rho_in,self.size_rho_in), np.complex128)
    mask_rho_tmp = self.mask_rho.reshape(self.nk,self.nb*self.nb)
    for ik in range(self.nk):
      Lvtmp = np.zeros((self.nb,self.nb,self.nb,self.nb), np.complex128)
      for b in range(self.nb):
        Lvtmp[:,b,:,b] = 0.5 * self.vmat_full[self.dirv,ik]
      for a in range(self.nb):
        Lvtmp[a,:,a,:] = Lvtmp[a,:,a,:] + 0.5 * self.vmat_full[self.dirv,ik]
      Lvtmp2 = Lvtmp.reshape(self.nb*self.nb, self.nb*self.nb)[:,mask_rho_tmp[ik]][mask_rho_tmp[ik],:]
      Lv[self.ind_k_rho[ik]:self.ind_k_rho[ik+1], self.ind_k_rho[ik]:self.ind_k_rho[ik+1]] = Lvtmp2
    return Lv
    '''
    Lv = np.zeros((self.nk,self.nb,self.nb,self.nk,self.nb,self.nb), np.complex128)
    for ik in range(self.nk):
      for b in range(self.nb):
        Lv[ik,:,b,ik,:,b] = 0.5 * self.vmat_full[self.dirv,ik]
      for a in range(self.nb):
        Lv[ik,a,:,ik,a,:] = Lv[ik,a,:,ik,a,:] + 0.5 * self.vmat_full[self.dirv,ik].T
    return Lv.reshape(self.nk*self.nb*self.nb,self.nk*self.nb*self.nb)[:,self.mask_rho][self.mask_rho,:]
    '''
  
  ##################################################
  # for ruling out states outside [emin, emax]
  ##################################################
  def reshape_and_trunc_outer_elec(self, trunc_outer):
    # reshape array for solving linearized master equation later
    self.Ek = self.Ek.reshape(-1)
    self.fk = self.fk.reshape(-1)
    self.dfdek = self.dfdek.reshape(-1)
    self.Ekmn_mat = self.Ekmn_mat.reshape(-1)
    self.dfde_mat = self.dfde_mat.reshape(-1)
    self.vmat = self.vmat.reshape(3,-1)
    self.vx = self.vx.reshape(-1)
    self.DfDk = self.DfDk.reshape(-1)
    self.mask_Ek = self.mask_Ek.reshape(-1)
    self.mask_rho = self.mask_rho.reshape(-1)
    self.valpol_mat = self.valpol_mat.reshape(self.valpol_mat.shape[0], self.nk*self.nb*self.nb)
    
    if trunc_outer and np.count_nonzero(self.mask_Ek) < self.nk*self.nb:
      print("\n--------------------------------------------------")
      print("rule out electronic states outside the energy window [emin, emax]")
      print("--------------------------------------------------")
      self.Ek = self.Ek[self.mask_Ek]
      self.fk = self.fk[self.mask_Ek]
      self.dfdek = self.dfdek[self.mask_Ek]
      self.Ekmn_mat = self.Ekmn_mat[self.mask_rho]
      self.ob_all = self.ob_all[:,self.mask_rho]
      self.rhopert_all = self.rhopert_all[:,self.mask_rho]
      self.valpol_mat = self.valpol_mat[:,self.mask_rho]
      self.dfde_mat = self.dfde_mat[self.mask_rho]
      self.vmat = self.vmat[:,self.mask_rho]
      self.vx = self.vx[self.mask_rho]
      self.DfDk = self.DfDk[self.mask_rho]
      
      self.dE = self.dE[self.mask_rho]
      #self.Lv = self.Lv[:,self.mask_rho][self.mask_rho,:]
      #if self.has_Bext:
      #  self.Lcoh = self.Lcoh[:,self.mask_rho][self.mask_rho,:]
    self.mask_bdiag = self.ob_all[0] != 0
    
    # scale and normalize ob; biorthogonalize rhopert to ob and normalize rhopert
    self.rhopert_all      = biorthogonalize(normalize_vector(self.ob_all[0:1]), self.rhopert_all,      True) # biorthogonalize rhopert to ob
    self.rhopert_all[1:4] = biorthogonalize(normalize_vector(self.ob_all[1:4]), self.rhopert_all[1:4], True) # biorthogonalize rhopert to ob
    self.rhopert_all[4:7] = biorthogonalize(normalize_vector(self.ob_all[4:7]), self.rhopert_all[4:7], True) # biorthogonalize rhopert to ob
    for ivp in range(self.latt.valpol_k.shape[0]):
      self.rhopert_all[((ivp+1)*7+0):((ivp+1)*7+7)] = biorthogonalize(normalize_vector(self.ob_all[((ivp+1)*7+0):((ivp+1)*7+1)]), self.rhopert_all[((ivp+1)*7+0):((ivp+1)*7+7)], True) # biorthogonalize rhopert to ob
      self.rhopert_all[((ivp+1)*7+1):((ivp+1)*7+4)] = biorthogonalize(normalize_vector(self.ob_all[((ivp+1)*7+1):((ivp+1)*7+4)]), self.rhopert_all[((ivp+1)*7+1):((ivp+1)*7+4)], True) # biorthogonalize rhopert to ob
      self.rhopert_all[((ivp+1)*7+4):((ivp+1)*7+7)] = biorthogonalize(normalize_vector(self.ob_all[((ivp+1)*7+4):((ivp+1)*7+7)]), self.rhopert_all[((ivp+1)*7+4):((ivp+1)*7+7)], True) # biorthogonalize rhopert to ob
    self.rhopert_all = normalize_vector(self.rhopert_all)
  
  def check_mask_rho(self):
    for ik in range(self.nk):
      for b1 in range(self.nb):
        in_range_1 = self.Ek[ik,b1] > self.emin and self.Ek[ik,b1] < self.emax
        if in_range_1 and not self.mask_Ek[ik,b1]:
          print("mask_Ek is not right")
          exit(1)
        if not in_range_1 and self.mask_Ek[ik,b1]:
          print("mask_Ek is not right")
          exit(1)
        for b2 in range(self.nb):
          in_range_2 = self.Ek[ik,b2] > self.emin and self.Ek[ik,b2] < self.emax
          if (in_range_1 and in_range_2) and not self.mask_rho[ik,b1,b2]:
            print("mask_rho is not right")
            exit(1)
          if (not in_range_1 or not in_range_2) and self.mask_rho[ik,b1,b2]:
            print("mask_rho is not right")
            exit(1)
    print("mask_Ek and mask_rho are right")
  
  def set_ind_state_rho(self, mask_Ek, mask_rho):
    ind_k_state = np.zeros(self.nk+1, np.int32)
    ind_k_rho = np.zeros(self.nk+1, np.int32)
    nb_k = np.count_nonzero(mask_Ek, axis=1) #np.zeros(self.nk, np.int32)
    print("nb_k = ",nb_k)
    for ik in range(self.nk+1):
      if ik < self.nk:
      	if nb_k[ik] != 0:
          inds = mask_Ek[ik].nonzero()[0]
          if nb_k[ik] != inds[-1]+1 - inds[0]:
            print("nb_k[",ik,"] is not right")
            exit(1)
          if nb_k[ik]*nb_k[ik] != np.count_nonzero(mask_rho[ik]):
            print("nb^2 != number of true of mask_rho[",ik,"]")
            exit(1)
      if ik > 0:
        ind_k_state[ik] = ind_k_state[ik-1] + nb_k[ik-1]
        ind_k_rho[ik]   = ind_k_rho[ik-1]   + nb_k[ik-1]*nb_k[ik-1]
    print("ind_k_state = ",ind_k_state)
    return ind_k_state, ind_k_rho, nb_k
  
  def set_mask_Ek_from_brange_inner(self):
    bStart_k = np.fromfile("brange_inner/bStart_k.bin", np.int32)
    bEnd_k = np.fromfile("brange_inner/bEnd_k.bin", np.int32)
    
    self.mask_Ek = np.zeros((self.nk, self.nb), dtype=bool)
    for ik in range(self.nk):
      self.mask_Ek[ik,bStart_k[ik]:bEnd_k[ik]] = True