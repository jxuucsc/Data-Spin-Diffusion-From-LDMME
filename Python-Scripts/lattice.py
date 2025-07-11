#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
sys.path.insert(1, '/home/xujq/codes/jdftx/tools/solve_linearDMME/')
import ldbd_files as ldbd
from help_solve_tl_fromldmme_ldbdfiles import *
import units
from timeit import default_timer as timer
t0 = timer()

class lattice(object):
  
  def init(self, valley = None):
    self.dim, self.R, self.cell_size = ldbd.read_ldbd_R()
    
    if valley is None:
      self.valley = np.array([])
    else:
      self.valley = valley
    if self.valley.shape[0] > 0:
      print("valley:\n",self.valley)
  
  def get_valley_k(self, k):
    nvp = (self.valley.shape[0]*(self.valley.shape[0] - 1))//2
    self.valpol_k = np.zeros((nvp, k.shape[0]), np.int8)
    if nvp < 1:
      return
    k2v = wrap(k[:,None] - self.valley[None,:])
    #print("k:\n", k[0:10])
    print("k to valley:\n", k2v[0:10])
    kcart2v = np.einsum("ab,cdb->cda", self.R, k2v)
    kl2v = np.linalg.norm(kcart2v, axis=2)
    self.valley_k = np.argmin(kl2v, axis=1)
    print("valley of k:\n", self.valley_k[0:10])
    if self.valley.shape[0] == 2:
      # consider two energy-degenerate valleys
      self.valpol_k[0] = 2*self.valley_k - 1
    print("valley pol. of k:\n",self.valpol_k[0,0:10])