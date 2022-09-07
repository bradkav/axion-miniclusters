import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import sys

sys.path.append('../')

import AMC
import mass_function



#print(np.log10(rho_init))

def calcM0(m_a):
    return 1e-11*(20e-6/m_a)**(1/2)

def freq_to_ma(f): #f in GHz
    return 8.27e-6*(f/2)  



fig, ax1 = plt.subplots(figsize=(8,6))

plt.xscale('log')
plt.yscale('log')

#header="B0 [G], Period [s], Misalignment angle [radians], NS Age [Myr], x [pc], y [pc], z [pc], AMC density [Msun/pc^3], AMC radius [pc], AMC mass [Msun],  Impact parameter [pc], Relative velocity [pc/s]"

#fname = "../../data/Interaction_params_PL_M_AMC_1.00e-14_M31_delta_1.txt.gz"
#rho, R, b, vrel = np.loadtxt(fname, usecols=(7, 8, 10, 11,), unpack=True)

#rho_peak = (rho/4)*(b/R)**(-9/4)

Mvals = ["16", "14", "12", "10", "08", "06"]

for i in range(6):
    fname = f"../../data/Interaction_params_PL_M_AMC_1.00e-{Mvals[i]}_M31_delta_1.txt.gz"
    rho_in, R, b, vrel = np.loadtxt(fname, usecols=(7, 8, 10, 11,), unpack=True)

    rho_init = mass_function.rho_of_delta(1)

    inds = rho_in < 1e30*rho_init
    rho = rho_in[inds]
    b = b[inds]
    R = R[inds]

    t_cross = 2*np.sqrt(R**2 - b**2)/vrel[inds]

    rho_peak = (rho/4)*(b/R)**(-9/4)
    bins = np.geomspace(1e-6*rho_init, 1e8*rho_init, 40)
    np.append(bins, 1e13)

    plt.hist(rho_peak, bins, alpha=1.0, label=r"$M_\mathrm{AMC} = 10^{-" + str(int(Mvals[i])) + "} \,M_\odot$", density=True, cumulative = True,   histtype='step', color='C' + str(int(i)))
    plt.axvline(rho_init, linestyle=':', color='grey', alpha=1, lw=1)
    
    print(Mvals[i], np.log10(np.mean(rho_peak)))
    
for i in range(6):
    fname = f"../../data/Interaction_params_PL_M_AMC_1.00e-{Mvals[i]}_M31_delta_10.txt.gz"
    rho_in, R, b, vrel = np.loadtxt(fname, usecols=(7, 8, 10, 11,), unpack=True)

    rho_init = mass_function.rho_of_delta(10)

    inds = rho_in < 1e30*rho_init
    rho = rho_in[inds]
    b = b[inds]
    R = R[inds]
    
    t_cross = 2*np.sqrt(R**2 - b**2)/vrel[inds]

    rho_peak = (rho/4)*(b/R)**(-9/4)
    bins = np.geomspace(1e-6*rho_init, 1e3*rho_init, 30)
    np.append(bins, 1e13)

    plt.hist(rho_peak, bins, alpha=0.7, density=True, cumulative = True,  histtype='step', color='C' + str(int(i)), linestyle='-')
    plt.axvline(rho_init, linestyle=':', color='grey', alpha=1, lw=1)
    print(Mvals[i], np.log10(np.mean(rho_peak)))
    
for i in range(6):
    fname = f"../../data/Interaction_params_PL_M_AMC_1.00e-{Mvals[i]}_M31_delta_30.txt.gz"
    rho_in, R, b, vrel = np.loadtxt(fname, usecols=(7, 8, 10, 11,), unpack=True)

    rho_init = mass_function.rho_of_delta(30)

    inds = rho_in < 1e30*rho_init
    rho = rho_in[inds]
    b = b[inds]
    R = R[inds]
    
    t_cross = 2*np.sqrt(R**2 - b**2)/vrel[inds]

    rho_peak = (rho/4)*(b/R)**(-9/4)
    #rho_surf = rho/4
    
    bins = np.geomspace(1e-6*rho_init, 1e3*rho_init, 30)
    #np.append(bins, 1e13)

    plt.hist(rho_peak, bins, alpha=0.5, density=True, cumulative = True,  histtype='step', color='C' + str(int(i)), linestyle='-')
    plt.axvline(rho_init, linestyle=':', color='grey', alpha=1, lw=1)
    print(Mvals[i], np.log10(np.mean(rho_peak)))
    
#plt.plot([-1e30, -1e30], alpha=1, color='k', linestyle='-', label='$\delta = 1$')
#plt.plot([-1e30, -1e30], alpha=1, color='k', linestyle='--', label='$\delta = 10$')
#plt.plot([-1e30, -1e30], alpha=1, color='k', linestyle=':', label='$\delta = 30$')
    

plt.text(1e2, 0.8, r"$\delta = 1$", ha='center', alpha=1.0, va='center', rotation=50, fontsize=16, color='k')
plt.text(1e6, 5e-2, r"$\delta = 10$", ha='center', alpha=0.7, va='center', rotation=50, fontsize=16, color='k')
plt.text(1e8, 5e-3, r"$\delta = 30$", ha='center', alpha=0.5, va='center', rotation=50, fontsize=16, color='k')
    
plt.xlim(1e-1, 1e12)
plt.ylim(1e-3, 5e1)
    
plt.xlabel(r"$\rho$ [$M_\odot/\mathrm{pc}^3$]")
plt.ylabel(r"$P(\rho_\mathrm{peak} < \rho)$")
    
plt.legend(loc='upper left', ncol=2, fontsize=14)


plt.savefig("../../plots/PeakDensityDistribution_cumulative.pdf", bbox_inches='tight')

plt.show()