import numpy as np
import sys

sys.path.append('../')

import dirs
import params

import mass_function
import Andromeda

from matplotlib import pyplot as plt

Msun = 1.0
pc = 1.0
s = 1.0

day = 3600*24*s

km = (1/3.086e+13)*pc
kpc = 1e3*pc

G_N = 4.302e-3*(pc/Msun)*(km/s)**2

M_NS = 1.4*Msun
R_NS = 11*km
vesc_NS = np.sqrt(2*G_N*M_NS/R_NS)

#n_NS = 1e-2*pc**-2

R_vals = np.geomspace(1, 10000, 1000)*pc

M_vals = np.geomspace(1e-18, 1e-6, 14)*Msun

Gamma = 0.0*M_vals


u = 270*km/s


fig, ax1 = plt.subplots(figsize=(8,6))
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color1 = 'C0'
color2 = 'C1'

IDstrs = ["_M31_delta_1", "_M31_delta_10", "_M31_delta_30"]
alphas = [1.0, 0.65, 0.3]
deltas = [1, 10, 30]

for j in [0, 1, 2]:
    IDstr = IDstrs[j]
    alpha = alphas[j]
    delta = deltas[j]

    #M_list_old, gamma_list_old, _, _, _, _ = np.loadtxt("../../data/MassGrid.txt", unpack=True)
    M_list, gamma_list, gamma_AScut_list, T_lower_list, T_med_list, T_upper_list = np.loadtxt("../../data/MassGrid" + IDstr + ".txt", unpack=True)

    #ax1.loglog(M_list_old, gamma_list_old*3600*24, linestyle=':', color=color, lw=2.0)
    ax1.loglog(M_list, gamma_list*3600*24, linestyle='-', color=color1, lw=2.0, alpha=alpha)
    #ax1.loglog(M_list, gamma_AScut_list*3600*24, linestyle='--', color=color)


    #ax2.fill_between(M_list, T_lower_list/(3600*24), T_upper_list/(3600*24), color=color, alpha=0.25)
    #ax2.loglog(M_list, T_med_list/(3600*24), color=color2, lw=2.0, alpha=alpha)

    #ax2.loglog(M_list, gamma_list*T_med_list, linestyle='--', color='k', alpha=alpha)

    rho_AMC = mass_function.rho_of_delta(delta)*Msun/pc**-3

    for i, M in enumerate(M_vals):
    
        r_AMC = (3*M/(4*np.pi*rho_AMC))**(1/3)
    
        sigma = np.pi*(r_AMC + R_NS*np.sqrt(1+vesc_NS**2/u**2))**2
    
        n_NS_list = Andromeda.nNS_CMZ(R_vals)*pc**-3
        n_AMC_list = Andromeda.rhoNFW(R_vals)*(Msun/pc**3)/M
        Gamma[i] = sigma*u*np.trapz(4*np.pi*R_vals**2*n_AMC_list*n_NS_list, R_vals)

    plt.loglog(M_vals, Gamma*day, alpha = alpha, color=color1, linestyle='--')


ax1.text(2e-19, 4e2, r"$\delta = 1$", ha='right', alpha = 1.0, va='center', fontsize=16, color='C0')
ax1.text(2e-19, 2e1, r"$\delta = 10$", ha='right', alpha=0.7, va='center', fontsize=16, color='C0')
ax1.text(2e-19, 2e0, r"$\delta = 30$", ha='right', alpha=0.4, va='center', fontsize=16, color='C0')

ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim(1e-21, 1e-4)

ax1.set_ylim(1e-5, 1e4)
#ax2.set_ylim(1e-5, 1e3)

ax1.tick_params(axis='x',rotation=45)
ax1.set_xticks(np.geomspace(1e-19, 1e-4, 16), minor=False)
ax1.set_yticks(np.geomspace(1e-5, 1e4, 10))



ax1.set_xlabel(r'$M_\mathrm{AMC}$ [$M_\odot$]', fontsize=16)
ax1.set_ylabel(r'Encounter rate $\Gamma_\mathrm{enc}$ [day$^{-1}$]', color=color1, fontsize=16)

#ax2.set_ylabel(r'Encounter time $T_\mathrm{enc}$ [day]', color=color2, fontsize=16) # we already handled the x-label with ax1

#ax2.tick_params(axis='y', labelcolor=color2)


#ax2.text(1e-6, 1e0, r"$\Gamma_\mathrm{enc} \times T_\mathrm{enc}$", ha='right', va='center', fontsize=16, color='k')

ax1.tick_params(axis='x',rotation=45)
ax1.set_xticks(np.geomspace(1e-19, 1e-4, 16), minor=False)

#ax2.text(1e-9, 2e4, r"M31, $f_\mathrm{AMC} = 100\%$")

plt.title(r"PL Density Profile; M31; $f_\mathrm{AMC} = 100\%$")

#ax1.text()

#ax2.axvspan(calcM0(freq_to_ma(12.3)), calcM0(freq_to_ma(7.8)))

plt.savefig("../../plots/EncounterRate_AMC_masses_delta_rough.pdf", bbox_inches='tight')



plt.show()
