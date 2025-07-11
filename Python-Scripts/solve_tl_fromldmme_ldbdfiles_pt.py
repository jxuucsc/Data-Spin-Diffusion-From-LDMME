#!/usr/bin/env python3
import numpy as np
import scipy as sci
import sys
import os
sys.path.insert(1, './')
sys.path.insert(1, '/home/xujq/codes/jdftx/tools/solve_linearDMME/')
from lattice import *
from electron import *
from electron_phonon import *
from solve_lifetimes import *
from solve_diffusion_lengths import *
#from approximate_lifetimess_diffusion_lengths_grbn import *
import solve_gevp_rgat as rgat
#import solve_gevp_rgat_left as rgat_left
import ldbd_files as ldbd
from help_solve_tl_fromldmme_ldbdfiles import *
from relax_realtime import *
from diffuse_realspace import *
import solve_tl_fromldmme_RayleighRitz as tlRR
import units
from timeit import default_timer as timer
t0 = timer()

Bext = np.array([0,0,0],np.float64) / units.Tesla # from Tesla to au
excess_density = None # excess carrier density (elec. - hole) in cm^-d
mu = 0 / units.eV # chemical potential or Fermi level, from eV to au
if_reset_erange = False
nkbt = 7
scale_sc = 1
scale_coh = 1
type_diff_eq = "Lpinv 1"
typeP = "lindblad"
#K = np.array([1./3, 1./3, 0], dtype=np.float64)
valley = None #np.array([K, -K])
ind_pert_sel = np.arange(0,4,dtype=np.int32) #np.array([0,1,2,3,7,8,9,10],np.int32)
ind_ob_sel = np.arange(0,4,dtype=np.int32) #np.array([0,1,2,3,7,8,9,10],np.int32)
renorm_spinpert = False
trunc_outer = True
Lsc_from = "dmd trunc" # "P", "dmd", "dmd trunc"
write_Lsc = True
write_eigen = True
orthnorm_URR = False
tmax = 0
xmax = 0

os.system("mkdir analysis_ldmme")
np.set_printoptions(precision=5)
##################################################
# read ldbd files and compute L
##################################################

latt = lattice()
latt.init(valley)
elec = electron()
elec.init(latt, excess_density, mu, scale_coh, Bext, renorm_spinpert, trunc_outer, if_reset_erange, nkbt)
eph = electron_phonon()
eph.init(latt, elec, scale_sc, typeP, trunc_outer, Lsc_from, write_Lsc)

t1 = timer()
print("\ntime for set up and some simple analysis: ",t1-t0)
sys.stdout.flush()
##################################################
# spin lifetimes via linear algebra
##################################################
print("\n##################################################")
print("solving spin lifetimes via linear algebra and analysis decay modes")
print("##################################################")
tau,e,UL,UR = solve_lifetimes_linalg(eph.L, "-L x = x e", elec.size_rho_in, write_eigen)

ob, ob_scale, rhopert = elec.select_reset_rhopert_and_ob(ind_pert_sel, ind_ob_sel)

print("\nAnalyse decay mode:")
VR_ob, VLH_rhopert, pert_and_ob = analyse_decay_mode(tau, UR, UL, 10, rhopert, ob_scale, elec.Ekmn_mat, elec.mu, elec.dfde_mat)
# spin relevance
spin_rel = np.linalg.norm(pert_and_ob[:,1:4,1:4].reshape(elec.nmode, 9), axis=1)
#spin_rel[1:3] = np.sqrt(np.mean(np.square(spin_rel[1:3])))
#spin_rel[1:4] = np.sqrt(np.mean(np.square(spin_rel[1:4])))
np.savetxt("analysis_ldmme/spin_relevance.dat", np.transpose([spin_rel[:], tau[:]*units.ps]))
# valley and valley spin relevance
if latt.valpol_k.shape[0] == 1:
  val_rel = np.linalg.norm(pert_and_ob[:,4:8,4:8].reshape(elec.nmode, 16), axis=1)
  np.savetxt("analysis_ldmme/valley_relevance.dat", np.transpose([val_rel[1:], tau[1:]*units.ps]))

# real-time dynamics
if tmax == 0:
  prod_srel_tau = spin_rel * tau
  mask1 = np.logical_and(tau < 1e15, spin_rel > 0.001 / elec.nk)
  mask = np.logical_and(mask1, prod_srel_tau > 1e-3*np.max(prod_srel_tau[mask1]))
  tau_srel = np.where(mask, tau, 0)
  tmax = np.max(tau_srel)
print("\ndo real-time dynamics with tmax (in ps) ",tmax*units.ps)
t = np.linspace(0,7*tmax,3000,dtype=np.float64)
Ot = relax_realtime(t, rhopert[1], ob[1], UL, UR, e, "analysis_ldmme/Sxt_sxpert", 0)
fit_Ot_expcos(t, Ot, "Sxt_sxpert")
Ot = relax_realtime(t, rhopert[1], ob[2], UL, UR, e, "analysis_ldmme/Syt_sxpert", 0)
Ot = relax_realtime(t, rhopert[1], ob[3], UL, UR, e, "analysis_ldmme/Szt_sxpert", 0)
Ot = relax_realtime(t, rhopert[2], ob[2], UL, UR, e, "analysis_ldmme/Syt_sypert", 0)
fit_Ot_expcos(t, Ot, "Syt_sypert")
Ot = relax_realtime(t, rhopert[3], ob[3], UL, UR, e, "analysis_ldmme/Szt_szpert", 0)
fit_Ot_expcos(t, Ot, "Szt_szpert")

'''
# print a certain eigenvector for wse2
norm_ULz = np.linalg.norm(UL[:,1])
norm_URz = np.linalg.norm(UR[:,1])
norm_dfde = np.linalg.norm(elec.dfde_mat)
np.savetxt("analysis_ldmme/UR1.dat", np.transpose([elec.Ekmn_mat*units.eV, elec.dfde_mat/norm_dfde, np.real(rhopert[1]), \
  np.real(UL[:,1]/norm_ULz), np.imag(UL[:,1]/norm_ULz), np.real(UR[:,1]/norm_URz), np.imag(UR[:,1]/norm_URz)]))
'''

'''
# print certain eigenvectors for gr-bn
UL_rotate,UR_rotate = get_U_rotate_within_deg(UL, UR, ob_scale, 1, 3)
UL_rotate = np.concatenate((UL_rotate, UL[:,3:4]), axis=1)
UR_rotate = np.concatenate((UR_rotate, UR[:,3:4]), axis=1)
Okb_UL,Okb_UR = compute_Ok_U(ob_scale, UL_rotate, UR_rotate, elec.mask_rho, elec.nk, elec.nb)

Ok_UL_c = np.sum(Okb_UL[:,:,:,2:4], axis=3)
Ok_UL_v = np.sum(Okb_UL[:,:,:,0:2], axis=3)
Ok_UR_c = np.sum(Okb_UR[:,:,:,2:4], axis=3)
Ok_UR_v = np.sum(Okb_UR[:,:,:,0:2], axis=3)
Ek_c = np.average(elec.Ek_full[:,2:4], axis=1)
Ek_v = np.average(elec.Ek_full[:,0:2], axis=1)
for m in range(3):
  outLc = np.concatenate((Ek_c.reshape(1,-1)*units.eV, np.real(Ok_UL_c[m]), np.imag(Ok_UL_c[m])))
  np.savetxt("analysis_ldmme/Okc_of_UL"+str(m)+".dat", np.transpose(outLc))
  outLv = np.concatenate((Ek_v.reshape(1,-1)*units.eV, np.real(Ok_UL_v[m]), np.imag(Ok_UL_v[m])))
  np.savetxt("analysis_ldmme/Okv_of_UL"+str(m)+".dat", np.transpose(outLv))
  outRc = np.concatenate((Ek_c.reshape(1,-1)*units.eV, np.real(Ok_UR_c[m]), np.imag(Ok_UR_c[m])))
  np.savetxt("analysis_ldmme/Okc_of_UR"+str(m)+".dat", np.transpose(outRc))
  outRv = np.concatenate((Ek_v.reshape(1,-1)*units.eV, np.real(Ok_UR_v[m]), np.imag(Ok_UR_v[m])))
  np.savetxt("analysis_ldmme/Okv_of_UR"+str(m)+".dat", np.transpose(outRv))
  
  out1c = np.concatenate((Ek_c.reshape(1,-1), np.abs(Ok_UR_c[m])))
  np.savetxt("analysis_ldmme/Okc_abs_of_UR"+str(m)+".dat", np.transpose(out1c))
  out2c = np.concatenate((Ek_c.reshape(1,-1), np.angle(Ok_UR_c[m])))
  np.savetxt("analysis_ldmme/Okc_angle_of_UR"+str(m)+".dat", np.transpose(out2c))
  out1v = np.concatenate((Ek_v.reshape(1,-1), np.abs(Ok_UR_v[m])))
  np.savetxt("analysis_ldmme/Okv_abs_of_UR"+str(m)+".dat", np.transpose(out1v))
  out2v = np.concatenate((Ek_v.reshape(1,-1), np.angle(Ok_UR_v[m])))
  np.savetxt("analysis_ldmme/Okv_angle_of_UR"+str(m)+".dat", np.transpose(out2v))
'''

'''
# print certain eigenvectors for gaas
# first rotate UR[:,1] and UR[:,2]
UL_rotate,UR_rotate = get_U_rotate_within_deg(UL, UR, ob_scale, 1, 4)
Okb_UL,Okb_UR = compute_Ok_U(ob_scale, UL_rotate, UR_rotate, elec.mask_rho, elec.nk, elec.nb)
Ok_UL = np.sum(Okb_UL, axis=3)
Ok_UR = np.sum(Okb_UR, axis=3)

Ek_avg = np.average(elec.Ek_full, axis=1) - np.min(elec.Ek)
np.savetxt("analysis_ldmme/dEk.dat", np.transpose([Ek_avg, elec.Ek_full[:,1] - elec.Ek_full[:,0]]))
for m in range(3):
  outL = np.concatenate((Ek_avg.reshape(1,-1)*units.eV, np.real(Ok_UL[m]), np.imag(Ok_UL[m])))
  np.savetxt("analysis_ldmme/Ok_of_UL"+str(m)+".dat", np.transpose(outL))
  outR = np.concatenate((Ek_avg.reshape(1,-1)*units.eV, np.real(Ok_UR[m]), np.imag(Ok_UR[m])))
  np.savetxt("analysis_ldmme/Ok_of_UR"+str(m)+".dat", np.transpose(outR))
'''

t2 = timer()
print("\ntime for spin lifetime: ",t2-t1)
sys.stdout.flush()

##################################################
# approximate method of spin lifetimes
##################################################
print("\n##################################################")
print("solving lifetimes approximately")
print("##################################################")
print("\n--------------------------------------------------")
print("Rayleigh-Ritz method (material dependent):")
print("--------------------------------------------------")

n_pert = rhopert.shape[0]
order = 10
UL_RR,UR_RR = setU_RayleighRitz_forLifetime(orthnorm_URR, ob_scale, rhopert, eph.L, order, elec.mask_rho, elec.nk, elec.nb)

tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UR_RR, UR_RR, np.arange(1,1+(n_pert-1)*1,dtype=np.int32), "", False, "smallest real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UL_RR, UR_RR, np.arange(1,1+(n_pert-1)*1,dtype=np.int32), "with Left", False, "smallest real abs")

# second-order, leading to reasonable results of long lifetimes
tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UR_RR, UR_RR, np.arange(1,1+(n_pert-1)*2,dtype=np.int32), "", False, "smallest real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UL_RR, UR_RR, np.arange(1,1+(n_pert-1)*2,dtype=np.int32), "with Left", False, "smallest real abs")

tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UR_RR, UR_RR, np.arange(1,1+(n_pert-1)*4,dtype=np.int32), "", False, "smallest real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UL_RR, UR_RR, np.arange(1,1+(n_pert-1)*4,dtype=np.int32), "with Left", False, "smallest real abs")

tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UR_RR, UR_RR, np.arange(1,1+(n_pert-1)*10,dtype=np.int32), "", False, "smallest real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UL_RR, UR_RR, np.arange(1,1+(n_pert-1)*10,dtype=np.int32), "with Left", False, "smallest real abs")
_,_,_ = tlRR.solve_tl_fromldmme_RayleighRitz_sevp("rate", -eph.L, UL_RR, UR_RR, np.arange(1,1+(n_pert-1)*10,dtype=np.int32), "with Left", False, "smallest real abs")

if order >= 20:
  tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UR_RR, UR_RR, np.arange(1,1+(n_pert-1)*20,dtype=np.int32), "", False, "smallest real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UL_RR, UR_RR, np.arange(1,1+(n_pert-1)*20,dtype=np.int32), "with Left", False, "smallest real abs")

if order >= 30:
  tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UR_RR, UR_RR, np.arange(1,1+(n_pert-1)*30,dtype=np.int32), "", False, "smallest real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UL_RR, UR_RR, np.arange(1,1+(n_pert-1)*30,dtype=np.int32), "with Left", False, "smallest real abs")

if order >= 40:
  tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UR_RR, UR_RR, np.arange(1,1+(n_pert-1)*40,dtype=np.int32), "", False, "smallest real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UL_RR, UR_RR, np.arange(1,1+(n_pert-1)*40,dtype=np.int32), "with Left", False, "smallest real abs")

if order >= 50:
  tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UR_RR, UR_RR, np.arange(1,1+(n_pert-1)*order,dtype=np.int32), "", False, "smallest real abs")
  tlRR.solve_tl_fromldmme_RayleighRitz("rate", -eph.L, None, UL_RR, UR_RR, np.arange(1,1+(n_pert-1)*order,dtype=np.int32), "with Left", False, "smallest real abs")
  _,_,_ = tlRR.solve_tl_fromldmme_RayleighRitz_sevp("rate", -eph.L, UL_RR, UR_RR, np.arange(0,1+(n_pert-1)*order,dtype=np.int32), "with Left", False, "smallest real abs")

t3 = timer()
print("\ntime for Rayleigh-Ritz lifetime: ",t3-t2)

##################################################
# material dependent analysis
##################################################
#approximate_lifetimes_grbn(UL_RR, UR_RR, eph, elec)

##################################################
# spin diffusion length
##################################################
print("\n##################################################")
print("solving spin diffusion length")
print("##################################################")

ldiff,lambda_l,UL_l,UR_l = solve_diffusion_lengths_linalg_driver(type_diff_eq, e, UL, UR, eph, elec, write_eigen, True)

print("\nAnalyse decay mode:")
VR_ob, VLH_rhopert, pert_and_ob = analyse_decay_mode(ldiff, UR_l, UL_l, 10, rhopert, ob_scale, elec.Ekmn_mat, elec.mu, elec.dfde_mat)
# spin relevance
spin_rel = np.linalg.norm(pert_and_ob[:,1:4,1:4].reshape(UR_l.shape[1], 9), axis=1)
#spin_rel[0:4] = np.sqrt(np.mean(np.square(spin_rel[0:4]))) # average over modes
#spin_rel[4:6] = np.sqrt(np.mean(np.square(spin_rel[4:6]))) # average over modes
#spin_rel[0:6] = np.sqrt(np.mean(np.square(spin_rel[0:6]))) # average over modes
np.savetxt("analysis_ldmme/spin_relevance_diffusion.dat", np.transpose([spin_rel, ldiff*units.meter*1e6]))

# real-space simulations
if xmax == 0:
  prod_srel_ldiff = spin_rel * np.abs(ldiff)
  mask1 = np.logical_and(np.abs(ldiff) < 1e15, spin_rel > 0.001 / elec.nk)
  mask = np.logical_and(mask1, prod_srel_ldiff > 1e-3*np.max(prod_srel_ldiff[mask1]))
  ldiff_srel = np.where(mask, np.abs(ldiff), 0)
  xmax = np.max(ldiff_srel)
print("\ndo real-space simulations with xmax (in micro-meter) ",xmax*units.meter*1e6)
x = np.linspace(0,7*xmax,3000,dtype=np.float64)
Ox = diffuse_realspace_1d(x, True, elec.rhopert_all[4], ob[1], UL_l, UR_l, lambda_l, "analysis_ldmme/Sxx_jxsxpert", 0)
fit_Ox_expcos(x, Ox, "Sxx_jxsxpert")
Ox = diffuse_realspace_1d(x, True, elec.rhopert_all[4], elec.ob_all[4], UL_l, UR_l, lambda_l, "analysis_ldmme/jxSx_jxsxpert", 0)
fit_Ox_expcos(x, Ox, "jxSx_jxsxpert")
Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_l, UR_l, lambda_l, "analysis_ldmme/Sxx_sxpert", 0)
fit_Ox_expcos(x, Ox, "Sxx_sxpert")
#Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[2], UL_l, UR_l, lambda_l, "analysis_ldmme/Syx_sxpert", 0)
#Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[3], UL_l, UR_l, lambda_l, "analysis_ldmme/Szx_sxpert", 0)
Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_l, UR_l, lambda_l, "analysis_ldmme/Sxx_sxpert_thrrel0.01", 0.01)
fit_Ox_expcos(x, Ox, "Sxx_sxpert_thrrel0.01")
Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_l, UR_l, lambda_l, "analysis_ldmme/Syx_sypert", 0)
fit_Ox_expcos(x, Ox, "Syx_sypert")
#Ox = diffuse_realspace_1d(x, True, elec.rhopert_all[5], ob[2], UL_l, UR_l, lambda_l, "analysis_ldmme/Syx_jxsypert", 0)
#fit_Ox_expcos(x, Ox, "Syx_jxsypert")
#Ox = diffuse_realspace_1d(x, True, elec.rhopert_all[5], elec.ob_all[5], UL_l, UR_l, lambda_l, "analysis_ldmme/jxSy_jxsypert", 0)
#fit_Ox_expcos(x, Ox, "jxSy_jxsypert")
#Ox = diffuse_realspace_1d(-x, False, rhopert[2], ob[2], UL_l, UR_l, lambda_l, "analysis_ldmme/Symx_sypert", 0)
Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_l, UR_l, lambda_l, "analysis_ldmme/Szx_szpert", 0)
fit_Ox_expcos(x, Ox, "Szx_szpert")
#Ox = diffuse_realspace_1d(-x, False, rhopert[3], ob[3], UL_l, UR_l, lambda_l, "analysis_ldmme/Szmx_szpert", 0)

sys.stdout.flush()
t4 = timer()
print("\ntime for diffusion length: ",t4-t3)
##################################################
# iterative method of spin diffusion length
##################################################
print("\n##################################################")
print("solving approximate diffusion lengths")
print("##################################################")
print("\n--------------------------------------------------")
print("Rayleigh-Ritz method:")
print("--------------------------------------------------")
#UL_l_RR, UR_l_RR = setU_RayleighRitz_forLength(orthnorm_URR, UL, UR, eph.L, elec.Lv, 3, elec.nmode)
# for Pt
UL_l_RR, UR_l_RR = setU_RayleighRitz_forLength(orthnorm_URR, ob_scale.T, rhopert.T, eph.L, elec.Lv, 3, elec.nmode)
UL_l_RR_2, UR_l_RR_2 = setU_RayleighRitz_forLength_2(orthnorm_URR, ob_scale.T, rhopert.T, eph.L, elec.Lv, 3, elec.nmode)

'''
# for gr-bn
# DD model for si
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([0,3],np.int32), "sx vsx", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([2,5],np.int32), "sz vsz", True, "smallest inv real abs")

tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,3],np.int32), "sx vsx, with Left", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([1,4],np.int32), "sy vsy, with Left", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([2,5],np.int32), "sz vsz, with Left", True, "smallest inv real abs")

# minimum model for sxy
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([0,1,3,4],np.int32), "sxy vsxy", True, "smallest inv real abs")

# minimum model for sxz
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([0,2,3,5],np.int32), "sxz vsxz", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,2,3,5],np.int32), "sxz vsxz, with Left", True, "smallest inv real abs")

# minimum model for s
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")

lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_sevp("lambda inv", -eph.Lpinv_Lv, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")
#lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")
#lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest inv real abs")
#lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions("lambda", -eph.L, elec.Lv, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest real abs")
#lambda_cDD,UL_cDD,UR_cDD = tlRR.solve_tl_fromldmme_RayleighRitz_solutions_g2s("lambda", -eph.L, elec.Lv, UL_l_RR, UR_l_RR, np.array([0,1,2,3,4,5],np.int32), "s vs, with Left", True, "smallest real abs")
print("\ndo real-space simulations using coupled-drift-diffusion model")
Ox = diffuse_realspace_1d(x, True, rhopert[1], ob[1], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Sxx_sxpert_cDD", 0)
fit_Ox_expcos(x, Ox, "Sxx_sxpert_cDD")
Ox = diffuse_realspace_1d(x, True, rhopert[2], ob[2], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Syx_sypert_cDD", 0)
fit_Ox_expcos(x, Ox, "Syx_sypert_cDD")
Ox = diffuse_realspace_1d(x, True, rhopert[3], ob[3], UL_cDD, UR_cDD, lambda_cDD, "analysis_ldmme/Szx_szpert_cDD", 0)
fit_Ox_expcos(x, Ox, "Szx_szpert_cDD")

t5_1 = timer()
print("time for Rayleigh-Ritz method: ",t5_1-t4)

# more diffusion length analysis
#approximate_diffusion_lengths_grbn(e[1:], UL[:,1:], UR[:,1:], ob_scale[1:], UL_RR, UR_RR, eph, elec)

t5_2 = timer()
print("time for more diffusion length analysis for gr-bn: ",t5_2-t5_1)
'''

'''
# for wse2
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", Lv, -L, UR_l_RR, UR_l_RR, np.array([0],np.int32), "sz", True, "smallest inv real abs")

tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", Lv, -L, UR_l_RR, UR_l_RR, np.array([0,1],np.int32), "sz vsz", True, "smallest inv real abs")
indV = np.array([0,1],np.int32)
AK = np.einsum("ba,bc->ac", UR_l_RR[:,indV].conj(), np.einsum("ab,bc->ac", -L, UR_l_RR[:,indV], optimize=True), optimize=True)
BK = np.einsum("ba,bc->ac", UR_l_RR[:,indV].conj(), np.einsum("ab,bc->ac", Lv, UR_l_RR[:,indV], optimize=True), optimize=True)
print("AK = \n",AK)
print("BK = \n",BK)

tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", Lv, -L, UL_l_RR, UR_l_RR, np.array([0,1],np.int32), "sz vsz, with Left", True, "smallest inv real abs")

tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", Lv, -L, UR_l_RR, UR_l_RR, np.array([0,1,2],np.int32), "sz vsz Lsz", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", Lv, -L, UR_l_RR, UR_l_RR, np.array([0,1,2,3],np.int32), "sz vsz Lsz Lvsz", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", Lv, -L, UL_l_RR, UR_l_RR, np.array([0,1,2,3],np.int32), "sz vsz Lsz Lvsz, with Left", True, "smallest inv real abs")
'''

# DD model for Pt
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([0,3],np.int32), "sx vsx", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([1,4],np.int32), "sy vsy", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UR_l_RR, UR_l_RR, np.array([2,5],np.int32), "sz vsz", True, "smallest inv real abs")

tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([0,3],np.int32), "sx vsx, with Left", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([1,4],np.int32), "sy vsy, with Left", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR, UR_l_RR, np.array([2,5],np.int32), "sz vsz, with Left", True, "smallest inv real abs")

tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([0,3],np.int32), "srx vsrx, slx vsrx", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([1,4],np.int32), "sry vsry, sly vsry", True, "smallest inv real abs")
tlRR.solve_tl_fromldmme_RayleighRitz("lambda inv", elec.Lv, -eph.L, UL_l_RR_2, UR_l_RR_2, np.array([2,5],np.int32), "srz vsrz, slz vsrz", True, "smallest inv real abs")

t5 = timer()
print("\ntime for approximate diffusion length: ",t5-t4)
print("total time: ",t5-t0)
