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
from timeit import default_timer as timer

class kmapper(object):
  
  def init(self, kmesh, k, U):
    print("\n##################################################")
    print("kmapper")
    print("##################################################")
    '''
    t0 = timer()
    
    #k = k - np.floor(k + 0.5)
    self.kmesh = kmesh
    self.mapping = self.create_mapping(k)
    self.do_test(k)
    self.set_neighbs(k)
    self.set_ovlp_nghb(U)
    
    t1 = timer()
    print("time for kmapper: ",t1-t0)
    '''

  def get_ikvec3(self, k):
    ikvec3 = np.zeros((k.shape[0],3),np.int32)
    for i in range(3):
      ikvec3[:,i] = np.round((k[:,i] % 1.0) * self.kmesh[i]).astype(np.int32) % self.kmesh[i]
    return ikvec3
  
  def create_mapping(self, k):
    mapping = {}
    ikvec3 = self.get_ikvec3(k)
    for idx in reversed(range(len(ikvec3))):
      key = tuple(ikvec3[idx])
      mapping[key] = idx
    return mapping
  
  def k2ik(self, k):
    if k.ndim != 2 or k.shape[1] != 3:
      raise ValueError("shape of k must be (-1, 3)")
    ikvec3 = self.get_ikvec3(k)
    return np.array([self.mapping.get(tuple(v3), -1) for v3 in ikvec3])
  
  def do_test(self, k):
    print(self.k2ik(k))
    inds = self.k2ik(k)
    for ik in range(k.shape[0]):
      if ik != inds[ik]:
        print("inds is wrong")
        exit(1)
  
  def set_neighbs(self, k):
    ikarr = np.arange(k.shape[0], dtype=np.int32)
    neighbs = np.zeros((k.shape[0],3,2), np.int32)
    central = np.zeros((k.shape[0],3), bool)
    dk = 1./self.kmesh
    for i in range(3):
      print("i = ",i)
      print("k:\n",k)
      km = k.copy()
      kp = k.copy()
      km[:,i] = km[:,i] - dk[i]
      print("km:\n",km[:,i])
      kp[:,i] = kp[:,i] + dk[i]
      print("kp:\n",kp[:,i])
      neighbs[:,i,0] = self.k2ik(km)
      has_left = neighbs[:,i,0] != -1
      neighbs[:,i,0] = np.where(has_left, neighbs[:,i,0], ikarr)
      print("left neighbs:\n",neighbs[:,i,0])
      for ik in range(k.shape[0]):
        if (neighbs[:,i,0] == ik):
          print("ik = ",ik)
          print("k = ",k[ik])
          print("km = ",km[ik])
        else:
          dist = np.norm(k[neighbs[:,i,0]] - km[ik])
          if dist >= 1e-6:
            print("ik = ",ik)
            print("k[ik] = ",k[ik])
            print("k[ikm] = ",k[neighbs[ik,i,0]])
            print("km = ",km[ik])
      neighbs[:,i,1] = self.k2ik(kp)
      has_right = neighbs[:,i,1] != -1
      neighbs[:,i,1] = np.where(has_right, neighbs[:,i,1], ikarr)
      print("right neighbs:\n",neighbs[:,i,0])
      for ik in range(k.shape[0]):
        if (neighbs[:,i,1] == ik):
          print("ik = ",ik)
          print("k = ",k[ik])
          print("kp = ",kp[ik])
        else:
          dist = np.norm(k[neighbs[:,i,1]] - kp[ik])
          if dist >= 1e-6:
            print("ik = ",ik)
            print("k[ik] = ",k[ik])
            print("k[ikp] = ",k[neighbs[ik,i,1]])
            print("kp = ",kp[ik])
      central[:,i] = np.logical_and(has_left, has_right)
      print("central:\n",central[:,i])
  
  def set_ovlp_nghb(self, U):
    o = np.zeros((U.shape[0],3,2,U.shape[2],U.shape[2]), np.complex128)
    for ik in range(U.shape[0]):
      for i in range(3):
        if neighbs[ik,i,0] == ik:
          o[ik,i,0] = np.eye(np.eye, dtype=np.complex128)
        else:
          o[ik,i,0] = np.einsum("kac,kcb->kab", U[ik].T.conj(), U[neighbs[ik,i,0]])
        if neighbs[ik,i,1] == ik:
          o[ik,i,1] = np.eye(np.eye, dtype=np.complex128)
        else:
          o[ik,i,1] = np.einsum("kac,kcb->kab", U[ik].T.conj(), U[neighbs[ik,i,1]])
    #for i in range(3):
    # o[:,i,0] = np.einsum("kac,kcb->kab", U.T.conj(), U[neighbs[:,i,0]])
    # o[:,i,1] = np.einsum("kac,kcb->kab", U.T.conj(), U[neighbs[:,i,1]])