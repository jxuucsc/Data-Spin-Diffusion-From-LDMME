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
from solve_diffusion_lengths import *
import solve_tl_fromldmme_RayleighRitz as tlRR
import units
from timeit import default_timer as timer

def approximate_lifetimes_grbn(UL_RR, UR_RR, eph, elec):
  print("\n##################################################")
  print("lifetime analysis for grbn:")
  print("##################################################")
  orig_stdout = sys.stdout
  f = open('approximate_lifetimes_grbn.out', 'w')
  sys.stdout = f
  
  sys.stdout = orig_stdout
  f.close()


def setU_RayleighRitz_forLength_2(orthnorm, UL, UR, L, Lv, n_pert, nmode):
  UR_l_RR = np.zeros((nmode,11*n_pert), dtype=np.complex128)
  UR_l_RR[:,0:n_pert]            =  UR[:,1:1+n_pert].copy()
  UR_l_RR[:,n_pert:2*n_pert]     =  UL[:,1:1+n_pert].copy()
  UR_l_RR[:,2*n_pert:3*n_pert]   =  np.einsum("ab,bm->am", Lv, UR[:,1:1+n_pert].copy())
  UR_l_RR[:,3*n_pert:4*n_pert]   =  np.einsum("ab,bm->am", Lv, UL[:,1:1+n_pert].copy())
  UR_l_RR[:,4*n_pert:5*n_pert]   = -np.einsum("ab,bm->am", L,  UL[:,1:1+n_pert].copy())
  UR_l_RR[:,5*n_pert:6*n_pert]   =  np.einsum("ab,bm->am", Lv, UR_l_RR[:,2*n_pert:3*n_pert])
  UR_l_RR[:,6*n_pert:7*n_pert]   =  np.einsum("ab,bm->am", Lv, UR_l_RR[:,3*n_pert:4*n_pert])
  UR_l_RR[:,7*n_pert:8*n_pert]   =  np.einsum("ab,bm->am", Lv, UR_l_RR[:,4*n_pert:5*n_pert])
  UR_l_RR[:,8*n_pert:9*n_pert]   = -np.einsum("ab,bm->am", L,  UR_l_RR[:,2*n_pert:3*n_pert])
  UR_l_RR[:,9*n_pert:10*n_pert]  = -np.einsum("ab,bm->am", L,  UR_l_RR[:,3*n_pert:4*n_pert])
  UR_l_RR[:,10*n_pert:11*n_pert] = -np.einsum("ab,bm->am", L,  UR_l_RR[:,4*n_pert:5*n_pert])
  #UR_l_RR[:,0:2*n_pert] = rgat.proj_out(UR_l_RR[:,0:2*n_pert], UR[:,0:1])
  UR_l_RR[:,0:11*n_pert] = np.transpose(normalize_vector(np.transpose(UR_l_RR[:,0:11*n_pert])))
  if orthnorm:
    UR_l_RR,_ = np.linalg.qr(UR_l_RR) # QR factorization to obtain orthonormal basis
  
  UL_l_RR = np.zeros((nmode,11*n_pert), dtype=np.complex128)
  UL_l_RR[:,0:n_pert]            =  UL[:,1:1+n_pert].copy()
  UL_l_RR[:,n_pert:2*n_pert]     =  UR[:,1:1+n_pert].copy()
  UL_l_RR[:,2*n_pert:3*n_pert]   =  np.einsum("ba,bm->am", Lv.conj(), UR[:,1:1+n_pert].copy())
  UL_l_RR[:,3*n_pert:4*n_pert]   =  np.einsum("ba,bm->am", Lv.conj(), UL[:,1:1+n_pert].copy())
  UL_l_RR[:,4*n_pert:5*n_pert]   = -np.einsum("ba,bm->am", L.conj(),  UR[:,1:1+n_pert].copy())
  UL_l_RR[:,5*n_pert:6*n_pert]   =  np.einsum("ba,bm->am", Lv.conj(), UL_l_RR[:,2*n_pert:3*n_pert])
  UL_l_RR[:,6*n_pert:7*n_pert]   =  np.einsum("ba,bm->am", Lv.conj(), UL_l_RR[:,3*n_pert:4*n_pert])
  UL_l_RR[:,7*n_pert:8*n_pert]   =  np.einsum("ba,bm->am", Lv.conj(), UL_l_RR[:,4*n_pert:5*n_pert])
  UL_l_RR[:,8*n_pert:9*n_pert]   = -np.einsum("ba,bm->am", L.conj(),  UL_l_RR[:,2*n_pert:3*n_pert])
  UL_l_RR[:,9*n_pert:10*n_pert]  = -np.einsum("ba,bm->am", L.conj(),  UL_l_RR[:,3*n_pert:4*n_pert])
  UL_l_RR[:,10*n_pert:11*n_pert] = -np.einsum("ba,bm->am", L.conj(),  UL_l_RR[:,4*n_pert:5*n_pert])
  #UL_l_RR[:,0:2*n_pert] = rgat.proj_out(UL_l_RR[:,0:2*n_pert], UL[:,0:1])
  UL_l_RR[:,0:11*n_pert] = np.transpose(normalize_vector(np.transpose(UL_l_RR[:,0:11*n_pert])))
  if orthnorm:
    UL_l_RR,_ = np.linalg.qr(UL_l_RR) # QR factorization to obtain orthonormal basis
  return UL_l_RR, UR_l_RR

def setU_RayleighRitz_forLength_3(orthnorm, UL, UR, Linv_Lv, n_pert, nmode):
  order = 3
  UR_l_RR = np.zeros((nmode,order*n_pert), dtype=np.complex128)
  UR_l_RR[:,0:n_pert]          =  UR[:,1:1+n_pert].copy()
  UR_l_RR[:,n_pert:2*n_pert]   = -np.einsum("ab,bm->am", Linv_Lv, UR_l_RR[:,0:n_pert])
  UR_l_RR[:,2*n_pert:3*n_pert] = -np.einsum("ab,bm->am", Linv_Lv, UR_l_RR[:,n_pert:2*n_pert])
  UR_l_RR[:,0:order*n_pert] = rgat.proj_out(UR_l_RR[:,0:order*n_pert], UR[:,0:1])
  UR_l_RR[:,0:order*n_pert] = np.transpose(normalize_vector(np.transpose(UR_l_RR[:,0:order*n_pert])))
  if orthnorm:
    UR_l_RR,_ = np.linalg.qr(UR_l_RR) # QR factorization to obtain orthonormal basis
  
  UL_l_RR = np.zeros((nmode,order*n_pert), dtype=np.complex128)
  UL_l_RR[:,0:n_pert]          =  UL[:,1:1+n_pert].copy()
  UL_l_RR[:,n_pert:2*n_pert]   = -np.einsum("ba,bm->am", Linv_Lv.conj(), UL_l_RR[:,0:n_pert])
  UL_l_RR[:,2*n_pert:3*n_pert] = -np.einsum("ba,bm->am", Linv_Lv.conj(), UL_l_RR[:,n_pert:2*n_pert])
  UL_l_RR[:,0:order*n_pert] = rgat.proj_out(UL_l_RR[:,0:order*n_pert], UL[:,0:1])
  UL_l_RR[:,0:order*n_pert] = np.transpose(normalize_vector(np.transpose(UL_l_RR[:,0:order*n_pert])))
  if orthnorm:
    UL_l_RR,_ = np.linalg.qr(UL_l_RR) # QR factorization to obtain orthonormal basis
  return UL_l_RR, UR_l_RR

def approximate_diffusion_lengths_grbn_aux(e, UL, UR, UL_RR, UR_RR, eph, elec):
  # drift-diffusion model for sz
  print("\ndrift-diffusion model for sz")
  UR_DD_z = np.zeros_like(UR[:,0:2])
  UR_DD_z[:,0] = UR[:,2]
  UR_DD_z[:,1] = np.einsum("ab,b->a", elec.Lv, UR[:,2], optimize=True)
  UL_DD_z = np.zeros_like(UR[:,0:2])
  UL_DD_z[:,0] = UL[:,2]
  UL_DD_z[:,1] = np.einsum("ab,b->a", elec.Lv, UL[:,2])
  AK = np.einsum("ba,bc->ac", UL_DD_z.conj(), np.einsum("ab,bc->ac", -eph.L, UR_DD_z, optimize=True), optimize=True)
  BK = np.einsum("ba,bc->ac", UL_DD_z.conj(), np.einsum("ab,bc->ac", elec.Lv, UR_DD_z, optimize=True), optimize=True)
  print("AK = \n",AK)
  print("BK = \n",BK)
  CK = np.einsum("ab,bc->ac", sci.linalg.inv(BK), AK)
  lambda_RR_z,_ = sci.linalg.eig(CK)
  print("ldiff_RR_z (from CK) in micro-meter = ", 0.529177e-4/np.real(lambda_RR_z))
  vx2_sz = np.einsum("a,a->", UL_DD_z[:,1].conj(), UR_DD_z[:,1])
  taupvsz_inv = np.einsum("a,a->", UL_DD_z[:,1].conj(), np.einsum("ab,b->a", -eph.L, UR_DD_z[:,1], optimize=True)) / vx2_sz
  print("vx2_sz = ", vx2_sz)
  print("taupvsz_inv = ", taupvsz_inv)
  print("vx2_sz * taupvsz_inv = ", vx2_sz * taupvsz_inv)
  lambda_DD_z = np.sqrt(taupvsz_inv * e[2] / vx2_sz)
  print("ldiff_DD_z in micro-meter = ", 0.529177e-4/np.real(lambda_DD_z))
  
  # drift-diffusion model for sx
  print("\ndrift-diffusion model for sx")
  UR_DD_x = np.zeros_like(UR[:,0:2])
  UR_DD_x[:,0] = UR[:,0]
  UR_DD_x[:,1] = np.einsum("ab,b->a", elec.Lv, UR[:,0], optimize=True)
  UL_DD_x = np.zeros_like(UR[:,0:2])
  UL_DD_x[:,0] = UL[:,0]
  UL_DD_x[:,1] = np.einsum("ab,b->a", elec.Lv, UL[:,0], optimize=True)
  AK = np.einsum("ba,bc->ac", UL_DD_x.conj(), np.einsum("ab,bc->ac", -eph.L, UR_DD_x, optimize=True), optimize=True)
  BK = np.einsum("ba,bc->ac", UL_DD_x.conj(), np.einsum("ab,bc->ac", elec.Lv, UR_DD_x, optimize=True), optimize=True)
  print("AK = \n",AK)
  print("BK = \n",BK)
  CK = np.einsum("ab,bc->ac", sci.linalg.inv(BK), AK, optimize=True)
  lambda_RR_x,_ = sci.linalg.eig(CK)
  print("ldiff_RR_x (from CK) in micro-meter = ", 0.529177e-4/np.real(lambda_RR_x))
  vx2_sx = np.einsum("a,a->", UL_DD_x[:,1].conj(), UR_DD_x[:,1])
  taupvsx_inv = np.einsum("a,a->", UL_DD_x[:,1].conj(), np.einsum("ab,b->a", -eph.L, UR_DD_x[:,1]), optimize=True) / vx2_sx
  print("vx2_sx = ", vx2_sx)
  print("taupvsx_inv = ", taupvsx_inv)
  print("vx2_sx * taupvsx_inv = ", vx2_sx * taupvsx_inv)
  lambda_DD_x = np.sqrt(taupvsx_inv * e[0] / vx2_sx)
  print("ldiff_DD_x in micro-meter = ", 0.529177e-4/np.real(lambda_DD_x))
  
  # drift-diffusion model for sy
  print("\ndrift-diffusion model for sy")
  UR_DD_y = np.zeros_like(UR[:,0:2])
  UR_DD_y[:,0] = UR[:,1]
  UR_DD_y[:,1] = np.einsum("ab,b->a", elec.Lv, UR[:,1], optimize=True)
  UL_DD_y = np.zeros_like(UR[:,0:2])
  UL_DD_y[:,0] = UL[:,1]
  UL_DD_y[:,1] = np.einsum("ab,b->a", elec.Lv, UL[:,1], optimize=True)
  AK = np.einsum("ba,bc->ac", UL_DD_y.conj(), np.einsum("ab,bc->ac", -eph.L, UR_DD_y, optimize=True), optimize=True)
  BK = np.einsum("ba,bc->ac", UL_DD_y.conj(), np.einsum("ab,bc->ac", elec.Lv, UR_DD_y, optimize=True), optimize=True)
  print("AK = \n",AK)
  print("BK = \n",BK)
  CK = np.einsum("ab,bc->ac", sci.linalg.inv(BK), AK, optimize=True)
  lambda_RR_y,_ = sci.linalg.eig(CK)
  print("ldiff_RR_y (from CK) in micro-meter = ", 0.529177e-4/np.real(lambda_RR_y))
  vx2_sy = np.einsum("a,a->", UL_DD_y[:,1].conj(), UR_DD_y[:,1])
  taupvsy_inv = np.einsum("a,a->", UL_DD_y[:,1].conj(), np.einsum("ab,b->a", -eph.L, UR_DD_y[:,1]), optimize=True) / vx2_sy
  print("vx2_sy = ", vx2_sy)
  print("taupvsy_inv = ", taupvsy_inv)
  print("vx2_sy * taupvsy_inv = ", vx2_sy * taupvsy_inv)
  lambda_DD_y = np.sqrt(taupvsy_inv * e[1] / vx2_sy)
  print("ldiff_DD_y in micro-meter = ", 0.529177e-4/np.real(lambda_DD_y))
  
  # sx-sz coupling
  print("\nsx-sz coupling")
  UL_xz = np.zeros((UL_RR.shape[0],4),np.complex128)
  UL_xz[:,0] = UL_DD_x[:,0]
  UL_xz[:,1] = UL_DD_z[:,0]
  UL_xz[:,2] = UL_DD_x[:,1]
  UL_xz[:,3] = UL_DD_z[:,1]
  UR_xz = np.zeros((UR_RR.shape[0],4),np.complex128)
  UR_xz[:,0] = UR_DD_x[:,0]
  UR_xz[:,1] = UR_DD_z[:,0]
  UR_xz[:,2] = UR_DD_x[:,1]
  UR_xz[:,3] = UR_DD_z[:,1]
  AK = np.einsum("ba,bc->ac", UL_xz.conj(), np.einsum("ab,bc->ac", -eph.L, UR_xz, optimize=True), optimize=True)
  BK = np.einsum("ba,bc->ac", UL_xz.conj(), np.einsum("ab,bc->ac", elec.Lv, UR_xz, optimize=True), optimize=True)
  print("AK = \n",AK)
  print("BK = \n",BK)
  CK = np.einsum("ab,bc->ac", sci.linalg.inv(BK), AK, optimize=True)
  print("CK = \n",CK)
  lambda_xz,_ = sci.linalg.eig(CK)
  print("ldiff_xz (from CK) in micro-meter = ", 1./np.real(lambda_xz)*0.529177e-4)
  print("T1xinv = ",e[0])
  print("T1zinv = ",e[2])
  print("vx2_sx * taupvsx_inv = ", vx2_sx * taupvsx_inv)
  print("vx2_sz * taupvsz_inv = ", vx2_sz * taupvsz_inv)
  vxz = np.einsum("a,a->", UL_DD_x[:,0].conj(), UR_DD_z[:,1])
  vzx = np.einsum("a,a->", UL_DD_z[:,0].conj(), UR_DD_x[:,1])
  print("vxz = ", vxz)
  print("vzx = ", vzx)
  vxz0 = np.einsum("a,a->", UL_RR[:,1].conj(), np.einsum("ab,b->a", elec.Lv, elec.dE*UR_RR[:,3], optimize=True)) / taupvsz_inv
  vzx0 = np.einsum("a,a->", UL_RR[:,3].conj(), np.einsum("ab,b->a", elec.Lv, elec.dE*UR_RR[:,1], optimize=True)) / taupvsx_inv
  print("vxz0 = ", vxz0)
  print("vzx0 = ", vzx0)
  print("T1xinv vxz = ", e[0] * vxz)
  print("T1zinv vzx = ", e[2] * vzx)
  print("T1zinv vxz = ", e[2] * vxz)
  print("T1xinv vzx = ", e[0] * vzx)
  AK_appr = np.zeros((4,4),np.complex128)
  AK_appr[0,0] = e[0]
  AK_appr[0,3] = 0 #e[0] * vxz
  AK_appr[1,1] = e[2]
  AK_appr[1,2] = 0 #e[2] * vzx
  AK_appr[2,1] = 0 #e[2] * vxz
  AK_appr[2,2] = taupvsx_inv * vx2_sx
  AK_appr[3,0] = 0 #e[0] * vzx
  AK_appr[3,3] = taupvsz_inv * vx2_sz
  print("AK_appr = \n",AK_appr)
  BK_appr = np.zeros((4,4),np.complex128)
  BK_appr[0,1] = vxz
  BK_appr[0,2] = vx2_sx
  BK_appr[1,0] = vzx
  BK_appr[1,3] = vx2_sz
  BK_appr[2,0] = vx2_sx
  BK_appr[3,1] = vx2_sz
  print("BK_appr = \n",BK_appr)
  print("1 / vx2_sx = ",1 / vx2_sx)
  print("1 / vx2_sz = ",1 / vx2_sz)
  print("vxz / vx2_sx / vx2_sz = ",vxz / vx2_sx / vx2_sz)
  print("vzx / vx2_sx / vx2_sz = ",vzx / vx2_sx / vx2_sz)
  print("BK_appr_inv = \n",sci.linalg.inv(BK_appr))
  CK_appr = np.einsum("ab,bc->ac", sci.linalg.inv(BK_appr), AK_appr, optimize=True)
  print("CK_appr = \n",CK_appr)
  lambda_appr_xz,_ = sci.linalg.eig(CK_appr)
  print("lambda_appr_xz = ",lambda_appr_xz)
  print("ldiff_xz (from CK_appr) in micro-meter = ", 1./np.real(lambda_appr_xz)*0.529177e-4)
  ctmpx = taupvsx_inv / vx2_sx
  ctmpz = taupvsz_inv / vx2_sz
  b = -(ctmpx*e[0] + ctmpz*e[2] + ctmpx*ctmpz*vxz*vzx)
  c = ctmpx*e[0] * ctmpz*e[2]
  print("b = ",b)
  print("c = ",c)
  print("b*b - 4*c = ",b*b - 4*c)
  print("sqrt(b*b - 4*c) = ",np.sqrt(b*b - 4*c))
  print("E2_plus = ", (-b+np.sqrt(b*b - 4*c))/2)
  print("E2_minus = ",(-b-np.sqrt(b*b - 4*c))/2)
  
  # sy-sz coupling
  print("\nsy-sz coupling")
  UL_yz = np.zeros((UL_RR.shape[0],4),np.complex128)
  UL_yz[:,0] = UL_DD_y[:,0]
  UL_yz[:,1] = UL_DD_z[:,0]
  UL_yz[:,2] = UL_DD_y[:,1]
  UL_yz[:,3] = UL_DD_z[:,1]
  UR_yz = np.zeros((UR_RR.shape[0],4),np.complex128)
  UR_yz[:,0] = UR_DD_y[:,0]
  UR_yz[:,1] = UR_DD_z[:,0]
  UR_yz[:,2] = UR_DD_y[:,1]
  UR_yz[:,3] = UR_DD_z[:,1]
  AK = np.einsum("ba,bc->ac", UL_yz.conj(), np.einsum("ab,bc->ac", -eph.L,  UR_yz, optimize=True), optimize=True)
  BK = np.einsum("ba,bc->ac", UL_yz.conj(), np.einsum("ab,bc->ac", elec.Lv, UR_yz, optimize=True), optimize=True)
  print("AK = \n",AK)
  print("BK = \n",BK)
  CK = np.einsum("ab,bc->ac", sci.linalg.inv(BK), AK, optimize=True)
  print("CK = \n",CK)
  lambda_yz,_ = sci.linalg.eig(CK)
  print("ldiff_yz (from CK) in micro-meter = ", 1./np.real(lambda_yz)*0.529177e-4)
  print("T1yinv = ",e[1])
  print("T1zinv = ",e[2])
  print("vx2_sy * taupvsy_inv = ", vx2_sy * taupvsy_inv)
  print("vx2_sz * taupvsz_inv = ", vx2_sz * taupvsz_inv)
  vyz = np.einsum("a,a->", UL_DD_y[:,0].conj(), UR_DD_z[:,1])
  vzy = np.einsum("a,a->", UL_DD_z[:,0].conj(), UR_DD_y[:,1])
  print("vyz = ", vyz)
  print("vzy = ", vzy)
  vyz0 = np.einsum("a,a->", UL_RR[:,2].conj(), np.einsum("ab,b->a", elec.Lv, elec.dE*UR_RR[:,3], optimize=True)) / taupvsz_inv
  vzy0 = np.einsum("a,a->", UL_RR[:,3].conj(), np.einsum("ab,b->a", elec.Lv, elec.dE*UR_RR[:,2], optimize=True)) / taupvsx_inv
  print("vyz0 = ", vyz0)
  print("vzy0 = ", vzy0)
  print("T1xinv vyz = ", e[0] * vyz)
  print("T1zinv vzy = ", e[2] * vzy)
  print("T1zinv vyz = ", e[2] * vyz)
  print("T1xinv vzy = ", e[0] * vzy)
  AK_appr = np.zeros((4,4),np.complex128)
  AK_appr[0,0] = e[1]
  AK_appr[0,3] = 0 #e[0] * vyz
  AK_appr[1,1] = e[2]
  AK_appr[1,2] = 0 #e[2] * vzy
  AK_appr[2,1] = 0 #e[2] * vyz
  AK_appr[2,2] = taupvsy_inv * vx2_sy
  AK_appr[3,0] = 0 #e[1] * vzy
  AK_appr[3,3] = taupvsz_inv * vx2_sz
  print("AK_appr = \n",AK_appr)
  BK_appr = np.zeros((4,4),np.complex128)
  BK_appr[0,1] = vyz
  BK_appr[0,2] = vx2_sy
  BK_appr[1,0] = vzy
  BK_appr[1,3] = vx2_sz
  BK_appr[2,0] = vx2_sy
  BK_appr[3,1] = vx2_sz
  print("BK_appr = \n",BK_appr)
  print("1 / vx2_sy = ",1 / vx2_sy)
  print("1 / vx2_sz = ",1 / vx2_sz)
  print("vyz / vx2_sy / vx2_sz = ",vyz / vx2_sy / vx2_sz)
  print("vzy / vx2_sy / vx2_sz = ",vzy / vx2_sy / vx2_sz)
  print("BK_appr_inv = \n",sci.linalg.inv(BK_appr))
  CK_appr = np.einsum("ab,bc->ac", sci.linalg.inv(BK_appr), AK_appr, optimize=True)
  print("CK_appr = \n",CK_appr)
  lambda_appr_yz,_ = sci.linalg.eig(CK_appr)
  print("lambda_appr_yz = ",lambda_appr_yz)
  print("ldiff_yz (from CK_appr) in micro-meter = ", 1./np.real(lambda_appr_yz)*0.529177e-4)
  ctmpy = taupvsy_inv / vx2_sy
  ctmpz = taupvsz_inv / vx2_sz
  b = -(ctmpy*e[1] + ctmpz*e[2] + ctmpy*ctmpz*vyz*vzy)
  c = ctmpy*e[1] * ctmpz*e[2]
  print("b = ",b)
  print("c = ",c)
  print("b*b - 4*c = ",b*b - 4*c)
  print("sqrt(b*b - 4*c) = ",np.sqrt(b*b - 4*c))
  print("E2_plus = ", (-b+np.sqrt(b*b - 4*c))/2)
  print("E2_minus = ",(-b-np.sqrt(b*b - 4*c))/2)
  
  # sx-sy coupling
  print("\nsx-sy coupling")
  UL_xy = np.zeros((UL_RR.shape[0],4),np.complex128)
  UL_xy[:,0] = UL_DD_x[:,0]
  UL_xy[:,1] = UL_DD_y[:,0]
  UL_xy[:,2] = UL_DD_x[:,1]
  UL_xy[:,3] = UL_DD_y[:,1]
  UR_xy = np.zeros((UL_RR.shape[0],4),np.complex128)
  UR_xy[:,0] = UR_DD_x[:,0]
  UR_xy[:,1] = UR_DD_y[:,0]
  UR_xy[:,2] = UR_DD_x[:,1]
  UR_xy[:,3] = UR_DD_y[:,1]
  AK = np.einsum("ba,bc->ac", UL_xy.conj(), np.einsum("ab,bc->ac", -eph.L, UR_xy, optimize=True), optimize=True)
  BK = np.einsum("ba,bc->ac", UL_xy.conj(), np.einsum("ab,bc->ac", elec.Lv, UR_xy, optimize=True), optimize=True)
  print("AK = \n",AK)
  print("BK = \n",BK)
  CK = np.einsum("ab,bc->ac", sci.linalg.inv(BK), AK, optimize=True)
  print("CK = \n",CK)
  lambda_xy,_ = sci.linalg.eig(CK)
  print("ldiff_xy (from CK) in micro-meter = ", 1./np.real(lambda_xy)*0.529177e-4)
  print("T1xinv = ",e[0])
  print("T1yinv = ",e[1])
  print("vx2_sx * taupvsx_inv = ", vx2_sx * taupvsx_inv)
  print("vx2_sy * taupvsz_inv = ", vx2_sy * taupvsz_inv)
  vxy = np.einsum("a,a->", UL_DD_x[:,0].conj(), UR_DD_y[:,1])
  vyx = np.einsum("a,a->", UL_DD_y[:,0].conj(), UR_DD_x[:,1])
  print("vxy = ", vxy)
  print("vyx = ", vyx)
  vxy0 = np.einsum("a,a->", UL_RR[:,1].conj(), np.einsum("ab,b->a", elec.Lv, elec.dE*UR_RR[:,2], optimize=True)) / taupvsz_inv
  vyx0 = np.einsum("a,a->", UL_RR[:,2].conj(), np.einsum("ab,b->a", elec.Lv, elec.dE*UR_RR[:,1], optimize=True)) / taupvsx_inv
  print("vxy0 = ", vxy0)
  print("vyx0 = ", vyx0)
  print("T1xinv vxy = ", e[0] * vxy)
  print("T1zinv vyx = ", e[1] * vyx)
  print("T1zinv vxy = ", e[1] * vxy)
  print("T1xinv vyx = ", e[0] * vyx)
  AK_appr = np.zeros((4,4),np.complex128)
  AK_appr[0,0] = e[0]
  AK_appr[0,3] = 0 #e[0] * vxy
  AK_appr[1,1] = e[1]
  AK_appr[1,2] = 0 #e[1] * vyx
  AK_appr[2,1] = 0 #e[1] * vxy
  AK_appr[2,2] = taupvsx_inv * vx2_sx
  AK_appr[3,0] = 0 #e[0] * vyx
  AK_appr[3,3] = taupvsy_inv * vx2_sy
  print("AK_appr = \n",AK_appr)
  BK_appr = np.zeros((4,4),np.complex128)
  BK_appr[0,1] = vxy
  BK_appr[0,2] = vx2_sx
  BK_appr[1,0] = vyx
  BK_appr[1,3] = vx2_sy
  BK_appr[2,0] = vx2_sx
  BK_appr[3,1] = vx2_sy
  print("BK_appr = \n",BK_appr)
  print("1 / vx2_sx = ",1 / vx2_sx)
  print("1 / vx2_sy = ",1 / vx2_sy)
  print("vxy / vx2_sx / vx2_sy = ",vxy / vx2_sx / vx2_sy)
  print("vyx / vx2_sx / vx2_sy = ",vyx / vx2_sx / vx2_sy)
  print("BK_appr_inv = \n",sci.linalg.inv(BK_appr))
  CK_appr = np.einsum("ab,bc->ac", sci.linalg.inv(BK_appr), AK_appr, optimize=True)
  print("CK_appr = \n",CK_appr)
  lambda_appr_xy,_ = sci.linalg.eig(CK_appr)
  print("lambda_appr_xy = ",lambda_appr_xy)
  print("ldiff_xy (from CK_appr) in micro-meter = ", 1./np.real(lambda_appr_xy)*0.529177e-4)
  ctmpx = taupvsx_inv / vx2_sx
  ctmpy = taupvsy_inv / vx2_sy
  b = -(ctmpx*e[0] + ctmpy*e[1] + ctmpx*ctmpy*vxy*vyx)
  c = ctmpx*e[0] * ctmpy*e[1]
  print("b = ",b)
  print("c = ",c)
  print("b*b - 4*c = ",b*b - 4*c)
  print("sqrt(b*b - 4*c) = ",np.sqrt(b*b - 4*c))
  print("E2_plus = ", (-b+np.sqrt(b*b - 4*c))/2)
  print("E2_minus = ",(-b-np.sqrt(b*b - 4*c))/2)
  
  # sx-sy-sz coupling
  print("\nsx-sy-sz coupling")
  UL_xyz = np.zeros((UL_RR.shape[0],6),np.complex128)
  UL_xyz[:,0] = UL_DD_x[:,0]
  UL_xyz[:,1] = UL_DD_y[:,0]
  UL_xyz[:,2] = UL_DD_z[:,0]
  UL_xyz[:,3] = UL_DD_x[:,1]
  UL_xyz[:,4] = UL_DD_y[:,1]
  UL_xyz[:,5] = UL_DD_z[:,1]
  UR_xyz = np.zeros((UL_RR.shape[0],6),np.complex128)
  UR_xyz[:,0] = UR_DD_x[:,0]
  UR_xyz[:,1] = UR_DD_y[:,0]
  UR_xyz[:,2] = UR_DD_z[:,0]
  UR_xyz[:,3] = UR_DD_x[:,1]
  UR_xyz[:,4] = UR_DD_y[:,1]
  UR_xyz[:,5] = UR_DD_z[:,1]
  AK = np.einsum("ba,bc->ac", UL_xyz.conj(), np.einsum("ab,bc->ac", -eph.L, UR_xyz, optimize=True), optimize=True)
  BK = np.einsum("ba,bc->ac", UL_xyz.conj(), np.einsum("ab,bc->ac", elec.Lv, UR_xyz, optimize=True), optimize=True)
  print("AK = \n",AK)
  print("BK = \n",BK)
  CK = np.einsum("ab,bc->ac", sci.linalg.inv(BK), AK, optimize=True)
  print("CK = \n",CK)
  lambda_xyz,_ = sci.linalg.eig(CK)
  print("ldiff_xyz (from CK) in micro-meter = ", 1./np.real(lambda_xyz)*0.529177e-4)
  vij = np.einsum("ba,bc->ac", UL_xyz[:,0:3].conj(), UR_xyz[:,3:6])
  print("vij = \n", vij)
  vij0 = np.einsum("ba,bc->ac", UL_RR[:,1:4].conj(), np.einsum("ab,b,bc->ac", elec.Lv, elec.dE, UR_RR[:,1:4], optimize=True)) / taupvsz_inv
  print("vij0 = ", vij0)
  AK_appr = np.zeros((6,6),np.complex128)
  AK_appr[0,0] = e[0]
  AK_appr[1,1] = e[1]
  AK_appr[2,2] = e[2]
  AK_appr[3,3] = taupvsx_inv * vx2_sx
  AK_appr[4,4] = taupvsy_inv * vx2_sy
  AK_appr[5,5] = taupvsz_inv * vx2_sz
  print("AK_appr = \n",AK_appr)
  BK_appr = np.zeros((6,6),np.complex128)
  BK_appr[0,1] = vij[0,1]
  BK_appr[0,2] = vij[0,2]
  BK_appr[1,0] = vij[1,0]
  BK_appr[1,2] = vij[1,2]
  BK_appr[2,0] = vij[2,0]
  BK_appr[2,1] = vij[2,1]
  BK_appr[0,3] = vx2_sx
  BK_appr[3,0] = vx2_sx
  BK_appr[1,4] = vx2_sy
  BK_appr[4,1] = vx2_sy
  BK_appr[2,5] = vx2_sz
  BK_appr[5,2] = vx2_sz
  print("BK_appr = \n",BK_appr)
  print("BK_appr_inv = \n",sci.linalg.inv(BK_appr))
  CK_appr = np.einsum("ab,bc->ac", sci.linalg.inv(BK_appr), AK_appr, optimize=True)
  print("CK_appr = \n",CK_appr)
  lambda_appr_xyz,_ = sci.linalg.eig(CK_appr)
  print("lambda_appr_xyz = ",lambda_appr_xyz)
  print("ldiff_xyz (from CK_appr) in micro-meter = ", 1./np.real(lambda_appr_xyz)*0.529177e-4)

def approximate_diffusion_lengths_grbn_more(e, UL, UR, eph, elec, orthnorm_URR):
  print("\n##################################################")
  print("more diffusion length analysis for gr-bn:")
  print("##################################################")
  orig_stdout = sys.stdout
  ftmp = open('approximate_diffusion_lengths_grbn_more.out', 'w')
  sys.stdout = ftmp
  
  UL_l_RR, UR_l_RR = setU_RayleighRitz_forLength(orthnorm_URR, UL, UR, eph.L, elec.Lv, 3, elec.nmode, False)
  approximate_diffusion_lengths_grbn_aux(e[1:], UL[:,1:], UR[:,1:], UL_l_RR, UR_l_RR, eph, elec)
  
  sys.stdout = orig_stdout
  ftmp.close()

def compute_Linv_appr(e, UL, UR, elec, eph):
  einv = np.where(np.real(e) != 0, 1./e, 0)
  Linv = np.einsum("am,bm->ab", np.einsum("am,m->am", UR[:,1:4], -einv[1:4], optimize=True), UL[:,1:4].conj(), optimize=True)
  Q = np.eye(elec.size_rho_in, dtype=np.complex128) - np.einsum("am,bm->ab", UR[:,0:4], UL[:,0:4].conj(), optimize=True)
  kernal = 1./np.diag(eph.Lsc)
  kernal[np.abs(kernal) > 1e15] = 0
  #tau = -np.einsum("am,am->m", UL[:,4:].conj(), np.einsum("a,am->am", kernal, UR[:,4:], optimize=True), optimize=True)
  #print("tau in ps:\n", tau*units.ps)
  #Linv = Linv + np.einsum("am,bm->ab", np.einsum("am,m->am", UR[:,4:], -tau, optimize=True), UL[:,4:].conj(), optimize=True)
  tau_avg = -np.dot(kernal, elec.dfde_mat) / np.sum(elec.dfde_mat)
  print("tau_avg in ps:\n", tau_avg*units.ps)
  Linv = Linv + np.einsum("am,bm->ab", UR[:,4:]*(-tau_avg), UL[:,4:].conj(), optimize=True)
  return Linv

def approximate_diffusion_lengths_grbn(e, UL, UR, UL_l_RR, UR_l_RR, eph, elec, orthnorm_URR, x, rhopert, ob):
  print("\n##################################################")
  print("diffusion length analysis for grbn based on RR method:")
  print("##################################################")
  orig_stdout = sys.stdout
  ftmp = open('approximate_diffusion_lengths_grbn_RR.out', 'w')
  sys.stdout = ftmp
  
  UL_l_RR_2, UR_l_RR_2 = setU_RayleighRitz_forLength_2(orthnorm_URR, UL, UR, eph.L, elec.Lv, 3, elec.nmode)
  
  Linv_appr = compute_Linv_appr(e, UL, UR, elec, eph)
  Linv_appr_Lv = np.einsum("ab,bc->ac", Linv_appr, elec.Lv)
  
  UL_l_RR_3, UR_l_RR_3 = setU_RayleighRitz_forLength_3(orthnorm_URR, UL, UR, eph.Lpinv_Lv, 3, elec.nmode)
  UL_l_RR_4, UR_l_RR_4 = setU_RayleighRitz_forLength_3(orthnorm_URR, UL, UR, Linv_appr_Lv, 3, elec.nmode)
  
  print("--------------------------------------------------")
  print("DD model:")
  print("--------------------------------------------------")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([0,3],np.int32), "sx vsx", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([2,5],np.int32), "sz vsz", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,3],np.int32), "sx vsx, with Left", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([1,4],np.int32), "sy vsy, with Left", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([2,5],np.int32), "sz vsz, with Left", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([0,6],np.int32), "srx vsrx, slx vsrx", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([1,7],np.int32), "sry vsry, sly vsry", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([2,8],np.int32), "srz vsrz, slz vsrz", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR, UR_l_RR, np.array([0,3],np.int32), "sx vsx, with Left, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR, UR_l_RR, np.array([1,4],np.int32), "sy vsy, with Left, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR, UR_l_RR, np.array([2,5],np.int32), "sz vsz, with Left, sevp", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_2, UR_l_RR_2, np.array([0,6],np.int32), "srx vsrx, slx vsrx, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_2, UR_l_RR_2, np.array([1,7],np.int32), "sry vsry, sly vsry, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_2, UR_l_RR_2, np.array([2,8],np.int32), "srz vsrz, slz vsrz, sevp", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_3, UR_l_RR_3, np.array([0,3],np.int32), "sx tvsx, with Left, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_3, UR_l_RR_3, np.array([1,4],np.int32), "sy tvsy, with Left, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_3, UR_l_RR_3, np.array([2,5],np.int32), "sz tvsz, with Left, sevp", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -Linv_appr_Lv, UL_l_RR_4, UR_l_RR_4, np.array([0,3],np.int32), "sx tvsx, with Left, appr sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -Linv_appr_Lv, UL_l_RR_4, UR_l_RR_4, np.array([1,4],np.int32), "sy tvsy, with Left, appr sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -Linv_appr_Lv, UL_l_RR_4, UR_l_RR_4, np.array([2,5],np.int32), "sz tvsz, with Left, appr sevp", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR_2, UR_l_RR_2, np.array([0,3,6,9],np.int32), "srx vsrx slx vslx", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR_2, UR_l_RR_2, np.array([1,4,7,10],np.int32), "sry vsry sly vsly", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR_2, UR_l_RR_2, np.array([2,5,8,12],np.int32), "srz vsrz slz vslz", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UR_l_RR_2, UR_l_RR_2, np.array([0,3,6,9],np.int32), "srx vsrx slx vslx, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UR_l_RR_2, UR_l_RR_2, np.array([1,4,7,10],np.int32), "sry vsry sly vsly, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UR_l_RR_2, UR_l_RR_2, np.array([2,5,8,12],np.int32), "srz vsrz slz vslz, sevp", True, "smallest inv real abs")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using DD model and GEVP (with srz vsrz, slz vsrz)")
  print("--------------------------------------------------")
  lambda_DD,UL_DD,UR_DD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([2,8],np.int32), "srz vsrz, slz vsrz", True, "smallest inv real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_DD, UR_DD, lambda_DD, "analysis_ldmme/Szx_szpert_DD", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_DD")
  
  print("--------------------------------------------------")
  print("ss-DD model with two directions:")
  print("--------------------------------------------------")
  # minimum model for sxy
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([0,1,3,4],np.int32), "sxy vsxy", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,1,3,4],np.int32), "sxy vsxy, with Left", True, "smallest inv real abs")
  
  # minimum model for syz
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([1,2,4,5],np.int32), "syz vsyz", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([1,2,4,5],np.int32), "syz vsyz, with Left", True, "smallest inv real abs")
  
  # minimum model for sxz
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([0,2,3,5],np.int32), "sxz vsxz", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,2,3,5],np.int32), "sxz vsxz, with Left", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([0,2,6,8],np.int32), "srxz vsrxz, slxz vsrxz", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR_2, UR_l_RR_2, np.array([0,2,3,5,6,8,9,12],np.int32), "srxz vsrxz slxz vslxz", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_2, UR_l_RR_2, np.array([0,2,6,8],np.int32), "srxz vsrxz, slxz vsrxz, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_2, UR_l_RR_2, np.array([0,2,3,5,6,8,9,12],np.int32), "srxz vsrxz slxz vslxz, sevp", True, "smallest inv real abs")
  
  print("--------------------------------------------------")
  print("ss-DD model:")
  print("--------------------------------------------------")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([0,1,2,6,7,8],np.int32), "sr vsr, sl vsr", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR_2, UR_l_RR_2, np.arange(0,12,dtype=np.int32), "sr vsr sl vsl", True, "smallest inv real abs")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using ss-DD model and SEVP (with sr vsr, sl vsl)")
  print("--------------------------------------------------")
  lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")
  #lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")
  #lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")
  #lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions("lambda", -eph.L, elec.Lv, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest real abs")
  #lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda", -eph.L, elec.Lv, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Sxx_sxpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Sxx_sxpert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Syx_sypert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Syx_sypert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Szx_szpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_cDD")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using ss-DD model and GEVP (with sr vsr, sl vsl)")
  print("--------------------------------------------------")
  lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Sxx_sxpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Sxx_sxpert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Syx_sypert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Syx_sypert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Szx_szpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_cDD")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using m-ss-DD model and SEVP (with sr vsr, sl vsr)")
  print("--------------------------------------------------")
  lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_2, UR_l_RR_2, np.array([0,1,2,6,7,8],np.int32), "sr vsr, sl, vsr", True, "smallest inv real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Sxx_sxpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Sxx_sxpert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Syx_sypert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Syx_sypert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Szx_szpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_cDD")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using m-ss-DD model and GEVP (with sr vsr, sl vsr)")
  print("--------------------------------------------------")
  lambda_rcDD,UL_rcDD,UR_rcDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([0,1,2,6,7,8],np.int32), "sr vsr, sl vsr", True, "smallest inv real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_rcDD, UR_rcDD, lambda_rcDD, "analysis_ldmme/Sxx_sxpert_rcDD", 0)
  fit_Ox_expcos(x, Ox, "Sxx_sxpert_rcDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_rcDD, UR_rcDD, lambda_rcDD, "analysis_ldmme/Syx_sypert_rcDD", 0)
  fit_Ox_expcos(x, Ox, "Syx_sypert_rcDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_rcDD, UR_rcDD, lambda_rcDD, "analysis_ldmme/Szx_szpert_rcDD", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_rcDD")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using ss-DD model with appr. Linv (with sr vsr, sl vsl)")
  print("--------------------------------------------------")
  
  lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -Linv_appr_Lv, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Sxx_sxpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Sxx_sxpert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Syx_sypert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Syx_sypert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Szx_szpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_cDD")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using ss-DD model with appr. Linv (with sr vsr, sl vsr)")
  print("--------------------------------------------------")
  lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -Linv_appr_Lv, UL_l_RR_2, UR_l_RR_2, np.array([0,1,2,6,7,8],np.int32), "sr vsr, sl, vsr", True, "smallest inv real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Sxx_sxpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Sxx_sxpert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Syx_sypert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Syx_sypert_cDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Szx_szpert_cDD", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_cDD")
  
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -Linv_appr_Lv, UR_l_RR_2, UR_l_RR_2, np.arange(0,12,dtype=np.int32), "sr vsr sl vsl, sevp", True, "smallest inv real abs")
  
  print("\n--------------------------------------------------")
  print("higher order RR:")
  print("--------------------------------------------------")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.arange(0,9,dtype=np.int32), "sr vsr Lvsr, with Left", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5,9,10,11],np.int32), "sr vsr vvsr, with Left", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.arange(0,12,dtype=np.int32), "order 1", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.arange(0,12,dtype=np.int32), "order 1, with Left", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.arange(0,24,dtype=np.int32), "order 2", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.arange(0,24,dtype=np.int32), "order 2, with Left", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR, UR_l_RR, np.arange(0,12,dtype=np.int32), "order 1, with Left, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR, UR_l_RR, np.arange(0,24,dtype=np.int32), "order 2, with Left, sevp", True, "smallest inv real abs")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using ss-DD model and GEVP (RR12)")
  print("--------------------------------------------------")
  lambda_RR12,UL_RR12,UR_RR12 = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.arange(0,12,dtype=np.int32), "order 1, with Left", True, "smallest inv real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_RR12, UR_RR12, lambda_RR12, "analysis_ldmme/Sxx_sxpert_RR12", 0)
  fit_Ox_expcos(x, Ox, "Sxx_sxpert_RR12")
  Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_RR12, UR_RR12, lambda_RR12, "analysis_ldmme/Syx_sypert_RR12", 0)
  fit_Ox_expcos(x, Ox, "Syx_sypert_RR12")
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_RR12, UR_RR12, lambda_RR12, "analysis_ldmme/Szx_szpert_RR12", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_RR12")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using ss-DD model and GEVP (RR24)")
  print("--------------------------------------------------")
  lambda_RR24,UL_RR24,UR_RR24 = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.arange(0,24,dtype=np.int32), "order 2, with Left", True, "smallest inv real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_RR24, UR_RR24, lambda_RR24, "analysis_ldmme/Sxx_sxpert_RR24", 0)
  fit_Ox_expcos(x, Ox, "Sxx_sxpert_RR24")
  Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_RR24, UR_RR24, lambda_RR24, "analysis_ldmme/Syx_sypert_RR24", 0)
  fit_Ox_expcos(x, Ox, "Syx_sypert_RR24")
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_RR24, UR_RR24, lambda_RR24, "analysis_ldmme/Szx_szpert_RR24", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_RR24")
  
  print("\n--------------------------------------------------")
  print("higher order RR (2nd set):")
  print("--------------------------------------------------")
  
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([0,1,2,6,7,8,9,10,11],np.int32), "sr vsr vsl, sl vsr vsl", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR_2, UR_l_RR_2, np.arange(0,12,dtype=np.int32), "sr vsr sl vsl", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([0,1,2,6,7,8,9,10,11,15,16,17,18,19,20,24,25,26,27,28,29],np.int32), "sr vsr vsl Lvsr/l vvsr/l, sr vsr vsl Lvsr/l vvsr/l", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([0,1,2,6,7,8,9,10,11,15,16,17,18,19,20,24,25,26,27,28,29],np.int32), "sr vsr vsl Lvsr/l vvsr/l, sl vsr vsl Lvsr/l vvsr/l", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR_2, UR_l_RR_2, np.array([0,1,2,3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,24,25,26,27,28,29],np.int32), "sr sl vsr vsl Lvsr/l vvsr/l", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([0,1,2,3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,24,25,26,27,28,29],np.int32), "sr sl vsr vsl Lvsr/l vvsr/l, sl sr vsr vsl Lvsr/l vvsr/l", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.arange(0,33,dtype=np.int32), "", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.arange(0,33,dtype=np.int32), "with Left", True, "smallest inv real abs")
  
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_2, UR_l_RR_2, np.array([0,1,2,6,7,8,9,10,11],np.int32), "sr vsr vsl, sl vsr vsl, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UR_l_RR_2, UR_l_RR_2, np.arange(0,12,dtype=np.int32), "sr vsr sl vsl, sevp", True, "smallest inv real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR_2, UR_l_RR_2, np.arange(0,33,dtype=np.int32), "with Left, sevp", True, "smallest inv real abs")
  
  print("\n--------------------------------------------------")
  print("real-space simulations using GEVP (with vkr = vkl = sr vsr sl vsl)")
  print("--------------------------------------------------")
  lambda_excDD,UL_excDD,UR_excDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda inv", elec.Lv, -eph.L, UR_l_RR_2, UR_l_RR_2, np.arange(0,12,dtype=np.int32), "sr vsr sl vsl", True, "smallest inv real abs")
  
  Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_excDD, UR_excDD, lambda_excDD, "analysis_ldmme/Sxx_sxpert_excDD", 0)
  fit_Ox_expcos(x, Ox, "Sxx_sxpert_excDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_excDD, UR_excDD, lambda_excDD, "analysis_ldmme/Syx_sypert_excDD", 0)
  fit_Ox_expcos(x, Ox, "Syx_sypert_excDD")
  Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_excDD, UR_excDD, lambda_excDD, "analysis_ldmme/Szx_szpert_excDD", 0)
  fit_Ox_expcos(x, Ox, "Szx_szpert_excDD")
  
  sys.stdout = orig_stdout
  ftmp.close()