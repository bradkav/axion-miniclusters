import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

from scipy.interpolate import interp1d

rc('text', usetex=True)
rc('font', size=20)

import sys
sys.path.append('../')

import dirs
import mass_function

def sciformat_1(x):
    n = np.floor(np.log10(x))
    m = x*(10.0**(-n))
    
    str1 = ""
    if (m > 1.001):
        str1 = r'$%.1f \times ' % (m,)
        str2 = r'10^{%d}$' % n
    else:
        str2 = r'$10^{%d}$' % n
    return str1 + str2

fstr = "_delta1.55"
Rstr = "8.00"
circstr = "_circ"

#print("> ------------------  CHANGE TO CIRC!!!!!!")

M_i, R_i, M_f, R_f = np.loadtxt(dirs.montecarlo_dir + "AMC_test%s_a=%s_NFW%s.txt"%(fstr,Rstr,circstr), delimiter = ", ", unpack=True, usecols=(0,1,3,4))
M_i_d, R_i_d, M_f_d, R_f_d = np.loadtxt(dirs.montecarlo_dir + "AMC_test%s_a=%s_NFWc10000%s.txt"%(fstr,Rstr,circstr), delimiter = ", ", unpack=True, usecols=(0,1,3,4))

rho_i = 3*M_i/(4*np.pi*R_i**3)
rho_f = 3*M_f/(4*np.pi*R_f**3)

rho_i_d = 3*M_i_d/(4*np.pi*R_i_d**3)
rho_f_d = 3*M_f_d/(4*np.pi*R_f_d**3)

print("Mean mass (c = 100):", np.mean(M_f[M_f > 1e-29]))
print("Mean mass (c = 10000):", np.mean(M_f_d[M_f_d > 1e-29]))

M_bins = np.geomspace(1e-15, 1e-9, 30)
R_bins = np.geomspace(1e-7, 1e-2, 30)
rho_bins = np.geomspace(1e-4, 1e5, 30)

M_c = M_bins[:-1] + np.diff(M_bins)/2
R_c = R_bins[:-1] + np.diff(R_bins)/2
rho_c = rho_bins[:-1] + np.diff(rho_bins)/2

P_Mf,_ = np.histogram(M_f[M_f > 1e-29], bins=M_bins, density=True)
P_Mf_d,_ = np.histogram(M_f_d[M_f_d > 1e-29], bins=M_bins, density=True)

P_Rf,_ = np.histogram(R_f[M_f > 1e-29], bins=R_bins, density=True)
P_Rf_d,_ = np.histogram(R_f_d[M_f_d > 1e-29], bins=R_bins, density=True)

P_rhof,_ = np.histogram(rho_f[M_f > 1e-29], bins=rho_bins, density=True)
P_rhof_d,_ = np.histogram(rho_f_d[M_f_d > 1e-29], bins=rho_bins, density=True)

#-------------------------------

def r_AS(M_AMC):
    m_22 = 2e-5/1e-22
    return 1e3*(1.6/m_22)*(M_AMC/1e9)**(-1/3)
M_char = 1e-16
R_char = 1.7e-6
rho_char = 3*M_char/(4*np.pi*R_char**3)

AMC_MF = mass_function.PowerLawMassFunction(2e-5, -0.7)
m_grid = np.geomspace(AMC_MF.mmin, AMC_MF.mmax, 1000)
w_grid = 0.0*m_grid
w_grid_d = 0.0*m_grid

nu = M_f/M_i
for i, m in enumerate(m_grid):
    frac = np.sum(rho_f < rho_char*nu*(m/M_char)**2)/len(rho_f)
    #print(frac)
    w_grid[i] = AMC_MF.dPdlogM(m)*frac
    
p_surv_AS = np.trapz(w_grid/m_grid, m_grid)
print(p_surv_AS)

nu = M_f_d/M_i_d
for i, m in enumerate(m_grid):
    frac = np.sum(rho_f_d < rho_char*nu*(m/M_char)**2)/len(rho_f_d)
    w_grid_d[i] = AMC_MF.dPdlogM(m)*frac
    
p_surv_AS_d = np.trapz(w_grid_d/m_grid, m_grid)
print(p_surv_AS_d)

#----------------------------------

xpad=8

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(20,5), gridspec_kw={'wspace': 0.25})
# fig.suptitle(r"$r_\mathrm{GC} = " + Rstr + "\,\mathrm{kpc}$, $p_\mathrm{surv} = %.3f$"%(p_surv,))


#-------------------------
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.axvline(1e-10, linestyle='--', color='k')
ax1.plot(M_c, M_c*P_Mf, color='C8', label=r'NFW ($c = 100$)')
ax1.plot(M_c, M_c*P_Mf_d, color='C9',label=r'NFW ($c = 10000$)')

#ax1.text(0.55, 0.9, "$a = 8\\,\\mathrm{kpc}$\n$p_\\mathrm{surv} = %.2f$"%(p_surv,),transform=ax1.transAxes, ha='left', va='top')
ax1.xaxis.set_tick_params(pad=xpad)

ax1.set_xlabel(r"$M_\mathrm{AMC}$ $[M_\odot]$")
# ax1.set_ylabel(r"$N_\mathrm{AMC}$")
ax1.set_ylabel(r"$M_\mathrm{AMC} \times P(M_\mathrm{AMC})$")

ax1.set_ylim(1e-4, 10)
ax1.set_xlim(1e-15, 1e-9)
ax1.legend(loc='upper left', fontsize=14)

ax1.text(0.05, 0.7, r"$r_\mathrm{GC} = 8 \,\mathrm{kpc}$",transform=ax1.transAxes, fontsize=15, color='k', ha='left')
ax1.text(0.05, 0.6, r"$p_\mathrm{surv}^{c=100} \,\,\,\, = $ " + sciformat_1(p_surv_AS),transform=ax1.transAxes, fontsize=15, color='C8', ha='left')
ax1.text(0.05, 0.5, r"$p_\mathrm{surv}^{c=10000} = $ " + sciformat_1(p_surv_AS_d),transform=ax1.transAxes, fontsize=15, color='C9', ha='left')


#--------------------------------
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.axvline(R_i[0], linestyle='--', color='C8')
ax2.axvline(R_i_d[0], linestyle='--', color='C9')
ax2.plot(R_c, R_c*P_Rf, color='C8')
ax2.plot(R_c, R_c*P_Rf_d, color='C9')


ax2.xaxis.set_tick_params(pad=xpad)

ax2.set_xlabel(r"$R_\mathrm{AMC}$ $[\mathrm{pc}]$")
ax2.set_ylabel(r"$R_\mathrm{AMC} \times P(R_\mathrm{AMC})$")

ax2.set_ylim(1e-4, 10)   
ax2.set_xlim(1e-7, 1e-1)


#---------------------
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.axvline(rho_i[0], linestyle='--', color='C8')
ax3.axvline(rho_i_d[0], linestyle='--', color='C9')
ax3.plot(rho_c, rho_c*P_rhof, color='C8')
ax3.plot(rho_c, rho_c*P_rhof_d, color='C9')

# ax3.hist(rho_i, alpha = a_plot, bins=np.geomspace(1e4, 1e10, 50))
# ax3.hist(rho_f, alpha = a_plot, bins=np.geomspace(1e4, 1e10, 50))

ax3.xaxis.set_tick_params(pad=xpad)
#ax3.yaxis.set_tick_params(pad=6)
ax3.set_ylabel(r"$\bar{\rho} \times P(\bar{\rho} )$")

ax3.set_xlabel(r"$\bar{\rho}$ $[M_\odot\,\mathrm{pc}^{-3}]$")
#ax3.set_ylim(1e-5, 0.9)

#ax3.plot([0], [0], ls='--', label='Unperturbed AMCs', color=cbefore)
#ax3.plot([0], [0], label='Perturbed AMCs (' + PROFILE + ')', color=cafter)
#ax3.legend(loc='best', fontsize=16)
ax3.set_ylim(1e-4, 10)
ax3.set_xlim(1e-6, 1e6)
#ax3.legend(loc='best')



plt.savefig("../../plots/Distributions_NFWtest.pdf", bbox_inches='tight')

plt.show()