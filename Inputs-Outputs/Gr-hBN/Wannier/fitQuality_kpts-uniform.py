#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
from numpy import linalg as LA
from scipy.interpolate import interp1d
import sys
import re
import os

eV = 1/27.21138505
kelvin2au = 1/3.157750248040761e5

kstart = 0 # index of first k point, same as in spin_dft.py
kend = 1292 # index of last k point, same as in spin_dft.py
nv = 2 # number of valence bands
nc = 2 # number of conduction bands
T = 300 * kelvin2au # temperature
mu = 0.1/27.211386 -0.162021 # chemical potential, here the same as the value given in JDFTx output totalE.eigStats or totalE.out, here inside band gap
degthr=1e-7
fname="bandstruct-uniform"

# Read totalE.out
for line in open("totalE.out"):
  if "nBands" in line:
    bCBM = int(round(float(re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", line)[0]))) # assume there is SOC
#print("number of bCBM: ",bCBM)
kpoints = np.loadtxt(fname+'.kpoints', skiprows=2, usecols=(1,2,3))
if kend > kstart:
  kpoints = kpoints[kstart:kend,:]
bstart = 0
bend = nv + nc

e_dft = np.loadtxt('../e_dft.dat')
if e_dft.shape[0] != kpoints.shape[0]:
  print("e_dft.shape[0] != kpoints.shape[0]")
  exit(1)
if e_dft.shape[1] != nv+nc:
  print("e_dft.shape[1] != nv+nc")
  exit(1)
de_v_dft = np.loadtxt('../de_dft.dat', usecols=(0,))
de_c_dft = np.loadtxt('../de_dft.dat', usecols=(1,))
b2z_dft = np.loadtxt('../b2z_dft.dat')
b2x_dft = np.loadtxt('../b2x_dft.dat')
b2y_dft = np.loadtxt('../b2y_dft.dat')
sz_diag_dft = np.loadtxt('../sz_diag_dft.dat')
sx_diag_dft = np.loadtxt('../sx_diag_dft.dat')
sy_diag_dft = np.loadtxt('../sy_diag_dft.dat')

#Get kfold and spin type from output file:
initDone = False
for line in open('totalE.out'):
  if line.startswith('Initialization completed'):
    initDone = True
  if (not initDone) and line.startswith('spintype'):
    spinKey = line.split()[1]
    mlwfType = (np.complex128 if (spinKey=="vector-spin" or spinKey=="spin-orbit") else np.float64)
  if (not initDone) and line.startswith('kpoint-folding'):
    kfold = np.array([int(tok) for tok in line.split()[1:4]])
kfoldProd = np.prod(kfold)
kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])

#Read the wannier cell map and weights:
cellMap = np.loadtxt("wannier.mlwfCellMap", usecols=[0,1,2]).astype(np.int32)
Wwannier = np.fromfile("wannier.mlwfCellWeights", dtype=np.float64)
nCells = cellMap.shape[0]
nBands = int(np.sqrt(Wwannier.shape[0] / nCells))
if bend > nBands:
  bend = nBands
  bstart = nBands - nc -nv
Wwannier = Wwannier.reshape((nCells,nBands,nBands)).swapaxes(1,2)

#Read and expand Wannier Hamiltonian, spin and gradient matrix elements:
iReduced = np.dot(np.mod(cellMap, kfold[None,:]), kStride)
Hwannier = Wwannier * np.fromfile("wannier.mlwfH", dtype=mlwfType).reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)[iReduced]
Swannier = Wwannier[:,None] * np.fromfile("wannier.mlwfS", dtype=mlwfType).reshape((kfoldProd,3,nBands,nBands)).swapaxes(2,3)[iReduced]
Lwannier = np.zeros(Swannier.shape)
if os.path.isfile("wannier.mlwfL"):
  #--- mlwfL stores r x [r,H] to keep real; multiply by -i to make r x p
  Lwannier = -1j * Wwannier[:,None] * np.fromfile("wannier.mlwfL", dtype=mlwfType).reshape((kfoldProd,3,nBands,nBands)).swapaxes(2,3)[iReduced]
Lintp = np.zeros((kpoints.shape[0],3,nBands,nBands))
if os.path.isfile("debug.L"):
  Lintp = np.fromfile("debug.L", np.complex128).reshape((-1,3,nBands,nBands)).swapaxes(2,3)
  

#Helper function that does the job of FeynWann, albeit much less efficiently:
def getEprops(k):
  #Fourier transform to k:
  phase = np.exp((2j*np.pi)*np.dot(k,cellMap.T))
  Hk = np.tensordot(phase, Hwannier, axes=1)
  Sk = np.tensordot(phase, Swannier,  axes=1)
  #Diagonalize and switch to eigen-basis:
  Ek,Vk = np.linalg.eigh(Hk) #Diagonalize
  Vk_trunc = Vk[:,bstart:bend]
  Sk_trunc = np.einsum("ba,ibc,cd->iad", Vk_trunc.conjugate(), Sk, Vk_trunc)
  return Ek, Sk_trunc

# degenerate-subspace of spin matrix
def deg(s, E):
  sdeg = np.zeros((E.shape[0], E.shape[0]), dtype=np.complex128)
  for i in range(E.shape[0]):
    for j in range(E.shape[0]):
      if np.abs(E[i] - E[j]) < degthr:
        sdeg[i,j] = s[i,j]
  return sdeg

def get_deg_range(bstart_deg, E):
  bend_deg = bstart_deg + 1
  while bend_deg < bend-bstart:
    if abs(E[bend_deg] - E[bstart_deg]) > degthr:
      break
    bend_deg = bend_deg + 1
  return bend_deg

def diagonalize_deg(m, E):
  eigs = np.zeros(E.shape[0],np.float64)
  U = np.zeros(m.shape,np.complex128)
  bstart_deg = 0
  while bstart_deg < bend-bstart:
    bend_deg = get_deg_range(bstart_deg, E)
    eigs[bstart_deg:bend_deg], U[bstart_deg:bend_deg,bstart_deg:bend_deg] = LA.eigh(m[bstart_deg:bend_deg,bstart_deg:bend_deg])
    #if bend_deg - bstart_deg >= 2:
    #  print("bstart_deg = ",bstart_deg," bend_deg = ",bend_deg)
    #  print("m: ", m)
    #  print("eigs: ",eigs)
    bstart_deg = bend_deg
  return eigs, U

# 2 * spin mixing = 1 - sqrt(sdeg^2)
def b2(s, E):
  eigs,_ = diagonalize_deg(s,E)
  return 0.5 - 0.5*np.abs(eigs)
  #return 0.5*(1 - np.sqrt(np.einsum("ij,ji->i", deg(s,E), deg(s,E)))).real
def Bin2(sd, E):
  r = np.zeros(nv+nc)
  for b in range(0,nv+nc):
    if (b+bstart) % 2 == 0 and b < nv+nc-1:
      r[b] = np.power(sd[b]*(E[b+1]-E[b]), 2)
    if (b+bstart) % 2 == 1 and b > 0:
      r[b] = np.power(sd[b]*(E[b]-E[b-1]), 2)
  return r

def fermi(E): # fermi-dirac
  return 1. / (np.exp((E - mu)/T) + 1)
def fermiprime(E): # absolute value of fermi-dirac derivative without prefactor 1/(kB*T)
  fk = fermi(E)
  return np.multiply(fk, 1-fk)

def err_contrib(diff, dfde):
  return np.dot(np.power(diff, 2), dfde)

#Calcuated energy-resolved spin mixing and averaged spin mixing along x and z
eig = np.zeros((kpoints.shape[0], bend-bstart))
b2z = np.zeros((kpoints.shape[0], bend-bstart))
b2x = np.zeros((kpoints.shape[0], bend-bstart))
b2y = np.zeros((kpoints.shape[0], bend-bstart))
lz_diag = np.zeros((kpoints.shape[0], bend-bstart))
lx_diag = np.zeros((kpoints.shape[0], bend-bstart))
err_e = 0
err_de = 0
err_b2z = 0
err_b2x = 0
err_b2y = 0
err_Bin2z = 0
err_Bin2x = 0
err_Bin2y = 0
de_avg = 0
de_dft_avg = 0
b2x_avg = 0 # globally averaged spin mixing parameter |b^2| along x
b2y_avg = 0 # globally averaged spin mixing parameter |b^2| along x
b2z_avg = 0 # globally averaged spin mixing parameter |b^2| along z
b2x_dft_avg = 0
b2y_dft_avg = 0
b2z_dft_avg = 0
Bin2z_avg = 0
Bin2x_avg = 0
Bin2y_avg = 0
Bin2z_dft_avg = 0
Bin2x_dft_avg = 0
Bin2y_dft_avg = 0
sum_dfde = 0 # weight for averaging
sum_dfde_dft = 0
for ik in range(kpoints.shape[0]):
  #if ik % 100 == 0:
  #  print("ik=",ik)
  #  sys.stdout.flush()
  
  # Determine number of bands skipped in wannier fitting
  if ik == 0:
    ekfull,_ = getEprops(kpoints[ik])
    ek = ekfull[bstart:bend]
    err_ek_min = np.sqrt(np.sum(np.power(ek - e_dft[ik], 2)))
    bskip = bCBM - nv
    for b in range(max(bCBM+nc-nBands,0),bCBM-nv):
      if b != bCBM - nv:
        bstart = bCBM - b - nv
        bend = bCBM - b + nc
        ek = ekfull[bstart:bend]
        err_ek = np.sqrt(np.sum(np.power(ek - e_dft[ik], 2)))
        if err_ek < err_ek_min:
          bskip = b
          err_ek_min = err_ek
    #print("bskip = ",bskip)
    bstart = bCBM - bskip - nv
    bend = bCBM - bskip + nc
  
  ekfull, sk = getEprops(kpoints[ik])
  ek = ekfull[bstart:bend]
  lk = Lintp[ik,:,bstart:bend,bstart:bend]
  
  # compute errors
  eig[ik] = ek
  dfde = fermiprime(ek)
  dfde_dft = fermiprime(e_dft[ik])
  sum_dfde = sum_dfde + np.sum(dfde)
  sum_dfde_dft = sum_dfde_dft + np.sum(dfde_dft)
  b2x[ik] = b2(sk[0], ek)
  b2y[ik] = b2(sk[1], ek)
  b2z[ik] = b2(sk[2], ek)
  lx_diag[ik],_ = diagonalize_deg(lk[0], ek)
  lz_diag[ik],_ = diagonalize_deg(lk[2], ek)
  err_e = err_e + err_contrib(ek - e_dft[ik], dfde)
  de_v = ek[1] - ek[0]
  de_c = ek[3] - ek[2]
  err_de = err_de + err_contrib(de_v - de_v_dft[ik], (dfde[0]+dfde[1]))
  err_de = err_de + err_contrib(de_c - de_c_dft[ik], (dfde[2]+dfde[3]))
  err_b2z = err_b2z + err_contrib(b2z[ik] - b2z_dft[ik], dfde)
  err_b2x = err_b2x + err_contrib(b2x[ik] - b2x_dft[ik], dfde)
  err_b2y = err_b2y + err_contrib(b2y[ik] - b2y_dft[ik], dfde)
  de_avg = de_avg + (dfde[0]+dfde[1])*de_v + (dfde[2]+dfde[3])*de_c
  de_dft_avg = de_dft_avg + (dfde_dft[0]+dfde_dft[1])*de_v_dft[ik] + (dfde_dft[2]+dfde_dft[3])*de_c_dft[ik]
  b2x_avg = b2x_avg + np.dot(dfde, b2x[ik]) # weighted sum
  b2y_avg = b2y_avg + np.dot(dfde, b2y[ik]) # weighted sum
  b2z_avg = b2z_avg + np.dot(dfde, b2z[ik]) # weighted sum
  b2x_dft_avg = b2x_dft_avg + np.dot(dfde_dft, b2x_dft[ik])
  b2y_dft_avg = b2y_dft_avg + np.dot(dfde_dft, b2y_dft[ik])
  b2z_dft_avg = b2z_dft_avg + np.dot(dfde_dft, b2z_dft[ik])
  sx_diag = np.einsum("ii->i", sk[0]).real
  sy_diag = np.einsum("ii->i", sk[1]).real
  sz_diag = np.einsum("ii->i", sk[2]).real
  Bin2z = Bin2(sz_diag, ek)
  Bin2x = Bin2(sx_diag, ek)
  Bin2y = Bin2(sy_diag, ek)
  Bin2z_dft = Bin2(sz_diag_dft[ik], e_dft[ik])
  Bin2x_dft = Bin2(sx_diag_dft[ik], e_dft[ik])
  Bin2y_dft = Bin2(sy_diag_dft[ik], e_dft[ik])
  err_Bin2z = err_Bin2z + err_contrib(Bin2z - Bin2z_dft, dfde)
  err_Bin2x = err_Bin2x + err_contrib(Bin2x - Bin2x_dft, dfde)
  err_Bin2y = err_Bin2y + err_contrib(Bin2y - Bin2y_dft, dfde)
  Bin2z_avg = Bin2z_avg + np.dot(dfde, Bin2z)
  Bin2x_avg = Bin2x_avg + np.dot(dfde, Bin2x)
  Bin2y_avg = Bin2y_avg + np.dot(dfde, Bin2y)
  Bin2z_dft_avg = Bin2z_dft_avg + np.dot(dfde_dft, Bin2z_dft)
  Bin2x_dft_avg = Bin2x_dft_avg + np.dot(dfde_dft, Bin2x_dft)
  Bin2y_dft_avg = Bin2y_dft_avg + np.dot(dfde_dft, Bin2y_dft)

err_de_avg = np.sqrt(err_de / sum_dfde)
rel_err_b2z_avg = np.sqrt(err_b2z / sum_dfde)/(b2z_dft_avg/sum_dfde_dft)*100
rel_err_b2x_avg = np.sqrt(err_b2x / sum_dfde)/(b2x_dft_avg/sum_dfde_dft)*100
rel_err_b2y_avg = np.sqrt(err_b2y / sum_dfde)/(b2y_dft_avg/sum_dfde_dft)*100
rel_err_Bin2z_avg = np.sqrt(err_Bin2z / sum_dfde)/(Bin2z_dft_avg/sum_dfde_dft)*100
rel_err_Bin2x_avg = np.sqrt(err_Bin2x / sum_dfde)/(Bin2x_dft_avg/sum_dfde_dft)*100
rel_err_Bin2y_avg = np.sqrt(err_Bin2y / sum_dfde)/(Bin2y_dft_avg/sum_dfde_dft)*100
if err_de_avg > 1e-8:
  exit(1)
print("de = ", de_avg/sum_dfde," ( ",de_dft_avg/sum_dfde_dft," )")
print("|b|^2_z = %.4e (%.4e) |b|^2_x = %.4e (%.4e) |b|^2_y = %.4e (%.4e)" \
  % (b2z_avg/sum_dfde, b2z_dft_avg/sum_dfde_dft, b2x_avg/sum_dfde, b2x_dft_avg/sum_dfde_dft, b2y_avg/sum_dfde, b2y_dft_avg/sum_dfde_dft))
print("Bin2_z = %.4e (%.4e) Bin2_x = %.4e (%.4e) Bin2_y = %.4e (%.4e)" \
  % (Bin2z_avg/sum_dfde, Bin2z_dft_avg/sum_dfde_dft, Bin2x_avg/sum_dfde, Bin2x_dft_avg/sum_dfde_dft, Bin2y_avg/sum_dfde, Bin2y_dft_avg/sum_dfde_dft))
np.savetxt("b2z_wann.dat", b2z)
np.savetxt("b2x_wann.dat", b2x)
np.savetxt("b2y_wann.dat", b2y)
#np.savetxt("lz_diag_wann.dat", lz_diag)
#np.savetxt("lx_diag_wann.dat", lx_diag)
#print("tau_{s,zz} / tau_{s,xx} = ", b2x_avg / b2z_avg)
print("error of energy: ", np.sqrt(err_e / sum_dfde))
print("error of energy splitting: ", np.sqrt(err_de / sum_dfde), "(",np.sqrt(err_de / sum_dfde)/(de_dft_avg/sum_dfde_dft)*100,"%)")
print("error of b2z: ", np.sqrt(err_b2z / sum_dfde), "(",np.sqrt(err_b2z / sum_dfde)/(b2z_dft_avg/sum_dfde_dft)*100,"%)")
print("error of b2x: ", np.sqrt(err_b2x / sum_dfde), "(",np.sqrt(err_b2x / sum_dfde)/(b2x_dft_avg/sum_dfde_dft)*100,"%)")
print("error of b2y: ", np.sqrt(err_b2y / sum_dfde), "(",np.sqrt(err_b2y / sum_dfde)/(b2y_dft_avg/sum_dfde_dft)*100,"%)")
print("error of Bin2z: ", np.sqrt(err_Bin2z / sum_dfde), "(",np.sqrt(err_Bin2z / sum_dfde)/(Bin2z_dft_avg/sum_dfde_dft)*100,"%)")
print("error of Bin2x: ", np.sqrt(err_Bin2x / sum_dfde), "(",np.sqrt(err_Bin2x / sum_dfde)/(Bin2x_dft_avg/sum_dfde_dft)*100,"%)")
print("error of Bin2y: ", np.sqrt(err_Bin2y / sum_dfde), "(",np.sqrt(err_Bin2y / sum_dfde)/(Bin2y_dft_avg/sum_dfde_dft)*100,"%)")
np.savetxt('e_wannier.dat', eig)
if err_de_avg < 5e-9 and rel_err_b2z_avg < 6 and rel_err_b2x_avg < 6 and rel_err_b2y_avg < 6 \
  and rel_err_Bin2z_avg < 3 and rel_err_Bin2x_avg < 3 and rel_err_Bin2y_avg < 3:
  print("Promising setups!")
