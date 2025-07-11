#!/usr/bin/env python3
import numpy as np
import sys
import re
import os

def read_ldbd_R():
  fR = "ldbd_data/ldbd_R.dat"
  dim = np.loadtxt(fR, dtype=np.int32, usecols=(0,), max_rows=1)
  R = np.loadtxt(fR, dtype=np.float64, skiprows=1)
  print("dim = ",dim)
  print("R = \n",R)
  volume = np.linalg.det(R)
  print("volume = ",volume)
  if dim == 3:
    return dim, R, volume
  elif dim == 2:
    print("area = ",volume/R[2,2])
    return dim, R, volume/R[2,2]

def read_ldbd_size():
  lines = open("ldbd_data/ldbd_size.dat", "r").readlines()
  isMetal = bool(int(re.findall(r"\d+", lines[0].strip())[0]))
  has_e = bool(int(re.findall(r"\d+", lines[0].strip())[1]))
  has_h = bool(int(re.findall(r"\d+", lines[0].strip())[2]))
  sc_has_e = bool(int(re.findall(r"\d+", lines[0].strip())[3]))
  sc_has_h = bool(int(re.findall(r"\d+", lines[0].strip())[4]))
  #print("isMetal has_e has_h sc_has_e sc_has_h", isMetal, has_e, has_h, sc_has_e, sc_has_h)
  if not isMetal:
    if has_e != sc_has_e or has_h != sc_has_h or has_e == has_h:
      print("first line of ldbd_size.dat is not ok")
      exit(1)
  nb = int(re.findall(r"\d+", lines[1].strip())[0])
  nv = int(re.findall(r"\d+", lines[1].strip())[1])
  bbot_eph = int(re.findall(r"\d+", lines[1].strip())[4])
  btop_eph = int(re.findall(r"\d+", lines[1].strip())[5])
  if not isMetal and has_h:
    bbot_eph = int(re.findall(r"\d+", lines[1].strip())[6])
    btop_eph = int(re.findall(r"\d+", lines[1].strip())[7])
  if btop_eph - bbot_eph != nb:
    print("btop_eph - bbot_eph != nb is not allowed in current version")
    exit(1)
  nk_full = float(re.findall(r"\d+\.\d+e[+-]\d+", lines[2].strip())[0])
  nk = int(re.findall(r"\d+", lines[2].strip())[3])
  kmesh = np.zeros(3, np.int32)
  for i in range(3):
    kmesh[i] = int(re.findall(r"\d+", lines[2].strip())[4+i])
  nkpair = int(re.findall(r"\d+", lines[3].strip())[0])
  if not isMetal and has_h:
    nkpair = int(re.findall(r"\d+", lines[3].strip())[1])
  T = float(re.findall(r"\d+\.\d+e[+-]\d+", lines[5].strip())[0])
  emin = float(re.findall(r"[+-]?\d+\.\d+e[+-]\d+", lines[8].strip())[4])
  emax = float(re.findall(r"[+-]?\d+\.\d+e[+-]\d+", lines[8].strip())[5])
  print("isMetal = ",isMetal)
  if not isMetal:
    print("has_h = ",has_h)
  print("nb = ",nb)
  print("nk_full = ",nk_full," nk = ",nk," , kmesh = ",kmesh)
  print("nkpair = ",nkpair)
  print("T = ",T)
  print("emin = ",emin," emax = ",emax)
  return isMetal, has_h, nb, nv, nk_full, nk, kmesh, nkpair, T, emin, emax

def read_ldbd_kvec(nk):
  return np.fromfile("ldbd_data/ldbd_kvec.bin", np.float64).reshape(nk,3)

def read_ldbd_kpair(nkpair, nb = 0, nv = 0):
  kp = np.zeros((nkpair,2),np.uintp)
  f1 = "ldbd_data/ldbd_kpair_k1st.bin"
  f2 = "ldbd_data/ldbd_kpair_k2nd.bin"
  if not os.path.exists(f1):
    if nb != nv:
      print("for hole, nb must be = nv in current version")
      exit(1)
    f1 = "ldbd_data/ldbd_kpair_k1st_hole.bin"
    f2 = "ldbd_data/ldbd_kpair_k2nd_hole.bin"
  kp[:,0] = np.fromfile(f1, np.uintp)
  kp[:,1] = np.fromfile(f2, np.uintp)
  return kp

def read_ldbd_ek(nk,nb):
  return np.fromfile("ldbd_data/ldbd_ek.bin", np.float64).reshape(nk,nb)

def read_ldbd_smat(nk,nb):
  return np.fromfile("ldbd_data/ldbd_smat.bin", np.complex128).reshape(nk,3,nb,nb)

def read_ldbd_vmat(nk,nb):
  return np.fromfile("ldbd_data/ldbd_vmat.bin", np.complex128).reshape(nk,3,nb,nb)

def read_ldbd_P(nkp, nb, nk, kp, typeP="lindblad", nv = 0):
  P = np.zeros((nk,nk,np.power(nb,4)), np.complex128)
  f1 = "ldbd_data/ldbd_P1_"+typeP+".bin"
  f2 = "ldbd_data/ldbd_P2_"+typeP+".bin"
  if not os.path.exists(f1):
    if nb != nv:
      print("for hole, nb must be = nv in current version")
      exit(1)
    f1 = "ldbd_data/ldbd_P1_"+typeP+"_hole.bin"
    f2 = "ldbd_data/ldbd_P2_"+typeP+"_hole.bin"
  P1 = np.fromfile(f1, np.complex128).reshape(nkp,np.power(nb,4))
  P2 = np.fromfile(f2, np.complex128).reshape(nkp,np.power(nb,4))
  for ikp in range(nkp):
    ik = kp[ikp,0]
    jk = kp[ikp,1]
    P[ik,jk] = P1[ikp]
    if ik < jk:
      P[jk,ik] = P2[ikp].conj()
    if ik > jk:
      print("ik > jk")
      exit(1)
  return 2*np.pi*P.reshape(nk,nk,nb,nb,nb,nb)

def read_ldbd_Lsc(nkp, nb, nk, kp):
  L = np.zeros((nk,nk,nb,nb,nb,nb), np.complex128)
  f1 = "lsc_files/Lscij.bin"
  f2 = "lsc_files/Lscji.bin"
  Lij = np.fromfile(f1, np.complex128).reshape(nkp,nb,nb,nb,nb)
  Lji = np.fromfile(f2, np.complex128).reshape(nkp,nb,nb,nb,nb)
  for ikp in range(nkp):
    ik = kp[ikp,0]
    jk = kp[ikp,1]
    L[ik,jk] = Lij[ikp] + np.transpose(Lij[ikp], (1,0,3,2)).conj()
    if ik < jk:
      L[jk,ik] = Lji[ikp] + np.transpose(Lji[ikp], (1,0,3,2)).conj()
    if ik > jk:
      print("ik > jk")
      exit(1)
  return np.transpose(L.reshape(nk,nk,nb*nb,nb*nb), (0,2,1,3)).reshape(nk*nb*nb,nk*nb*nb)

def read_ldbd_Lsc_trunc(nkp, nb, nk, kp, size_rho_in, ind_k_rho, nb_k):
  L = np.zeros((size_rho_in,size_rho_in), np.complex128)
  f1 = "lsc_files/Lscij.bin"
  f2 = "lsc_files/Lscji.bin"
  Lij = np.fromfile(f1, np.complex128)
  Lji = np.fromfile(f2, np.complex128)
  ind_L = 0
  for ikp in range(nkp):
    ik = kp[ikp,0]
    jk = kp[ikp,1]
    nb_ik = nb_k[ik]
    nb_jk = nb_k[jk]
    nb_ik_sq = nb_ik**2
    nb_jk_sq = nb_jk**2
    ind_ik = ind_k_rho[ik]
    ind_ik_end = ind_ik + nb_ik_sq
    ind_jk = ind_k_rho[jk]
    ind_jk_end = ind_jk + nb_jk_sq
    Lsize = nb_ik_sq * nb_jk_sq
    Lijtmp = Lij[ind_L:ind_L+Lsize].reshape(nb_ik,nb_ik,nb_jk,nb_jk)
    L[ind_ik:ind_ik_end, ind_jk:ind_jk_end] = (Lijtmp + np.transpose(Lijtmp, (1,0,3,2)).conj()).reshape(nb_ik_sq, nb_jk_sq)
    if ik < jk:
      Ljitmp = Lji[ind_L:ind_L+Lsize].reshape(nb_jk,nb_jk,nb_ik,nb_ik)
      L[ind_jk:ind_jk_end, ind_ik:ind_ik_end] = (Ljitmp + np.transpose(Ljitmp, (1,0,3,2)).conj()).reshape(nb_jk_sq, nb_ik_sq)
    if ik > jk:
      print("ik > jk")
      exit(1)
    ind_L = ind_L + Lsize
  print("size of Lij = ", Lij.shape[0])
  print("size of L with relevant k pairs = ", ind_L)
  return L