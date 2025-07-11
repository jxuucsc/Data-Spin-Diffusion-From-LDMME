#!/usr/bin/env python3
import numpy as np
import scipy as sci
import os
from timeit import default_timer as timer

def sort_E_Y(type_e, E, YL, Y):
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
    return E[ind], Y[:,ind]
  else:
    return E[ind], YL[:,ind], Y[:,ind]

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

def proj_out(Q, V, opt_einsum = True):
  return Q - np.einsum("ab,bc->ac", V, np.einsum("ba,bc->ac", V.conj(), Q, optimize=opt_einsum), optimize=opt_einsum)

def check_solution(A, B, E, x, opt_einsum = True):
  err = np.max(np.abs(np.einsum("ab,bc->ac", A, x, optimize=opt_einsum) - np.einsum("ab,bc->ac", B, np.einsum("bc,c->bc", x, E, optimize=opt_einsum), optimize=opt_einsum)))
  print("err of Ax-BxE: ",err)

def rayleigh_ritz_method(A, B, V, type_e = "largest real", opt_einsum = True):
  Atilde = np.einsum("ba,bc->ac", V.conj(), np.einsum("ab,bc->ac", A, V, optimize=opt_einsum), optimize=opt_einsum)
  Btilde = np.einsum("ba,bc->ac", V.conj(), np.einsum("ab,bc->ac", B, V, optimize=opt_einsum), optimize=opt_einsum)
  E, Y = sci.linalg.eig(Atilde, b=Btilde)
  return sort_E_Y(type_e, E, None, Y)

def check_orthonormal(V, opt_einsum = True):
  err = np.max(np.abs(np.eye(V.shape[1], dtype=np.complex128) - np.einsum("ba,bc->ac", V.conj(), V, optimize=opt_einsum)))
  if err > 1e-6:
    print("error of I - V^H V: ",err)

def solve_gevp_rgat(A, B, l, p, V, type_e = "largest real", type_err = "rel", thr = 1e-4, nter_max = 100, opt_einsum = True):
  # solve a few extreme eigenvalues and associated right eigenvectors
  # of generlized eigenvalue problem: A x = B x E
  # based on "Algorithm 1" in paper
  # "A method for computing a few eigenpairs of large generalized eigenvalue problems."
  # Appl. Numer. Math. 183, 108-117 (2023). by M. Alkilayh, L. Reichel, and Q. Ye
  
  #initial check
  n = A.shape[0]
  if V.shape[1] != 2*p:
    print("V0 is a n*2p matrix")
    exit(1)
   
  t0 = timer()
  err = 2 * thr
  # step a, solve approximate eigenvalues and eigenvectors (Ritz values and vectors)
  E0, Y = rayleigh_ritz_method(A, B, V, type_e, opt_einsum)
  print("initial E: ",E0[0:l])
  print("other E: ",E0[l:2*p])
  Eold = E0.copy()
  E = np.zeros_like(E0)
  
  for it in range(nter_max):
    # step b
    Q = np.einsum("ab,bc->ac", V, Y[:,0:p], optimize=opt_einsum) # Q is r in Appl. Numer. Math. 183, 108-117 (2023)
    Q = np.einsum("ab,bc->ac", A, Q, optimize=opt_einsum) - np.einsum("ab,bc->ac", B, np.einsum("bc,c->bc",Q,Eold[0:p],optimize=opt_einsum), optimize=opt_einsum)
    # step c, orthonormal basis of r
    Q,_ = np.linalg.qr(Q)
    
    # step d, othronormal basis of Y[:,0:p]
    Vtilde,_ = np.linalg.qr(Y[:,0:p])
    # step e
    V[:,0:p] = np.einsum("ab,bc->ac", V, Vtilde)
    del Vtilde
    
    # step f and step j, new Krylov subspace
    V[:,p:2*p] = proj_out(Q, V[:,0:p]) # (I - V V^H) Q
    del Q
    normQ = np.linalg.norm(V[:,p:2*p], axis=0)
    V[:,p:2*p] = np.einsum("ab,b->ab", V[:,p:2*p], 1./normQ)
        
    # step k, solve new eigenvalues and eigenvectors (Ritz values and vectors)
    E, Y = rayleigh_ritz_method(A, B, V, type_e, opt_einsum)
    
    err = compute_error(type_e, E[0:l], Eold[0:l], type_err, thr)
    if err < thr:
      break
    else:
      Eold = E
  
  # final eigenvalues and (right) eigenvectors
  print("number of iterations = ",it," with error = ",err)
  t1 = timer()
  print("time = ",t1-t0)
  x = np.einsum("ab,bc->ac", V, Y[:,0:l])
  del Y
  print("\nfinal E: ",E[0:l])
  print("other E: ",E[l:2*p])
  print("norm of x: ", np.linalg.norm(x, axis=0))
  check_solution(A, B, E[0:l], x)
  return E0, E[0:l], x