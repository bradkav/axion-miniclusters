import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

import re
import argparse

rc('text', usetex=True)
rc('font', size=18)

root_dir = '../../data/'
import sys
sys.path.append('../')
import dirs
dist_dir = dirs.data_dir + "distributions/"


parser = argparse.ArgumentParser(description='...')
parser.add_argument('-R', '--R', help='Galactocentric radius in kpc', type=float, required=True)
parser.add_argument('-profile','--profile', help='Density profile for AMCs', type=str, default="PL")
args = parser.parse_args()

profile = args.profile
R = args.R
Rstr = f'{R:.2f}'

    
IDstr = ""

cut_text = ""

M_i, P_M_i = np.loadtxt(dist_dir + "distribution_mass_" + profile + "_unperturbed%s.txt"%(IDstr), unpack=True, delimiter=',')
M_f, P_M_f = np.loadtxt(dist_dir + "distribution_mass_" + Rstr + "_" + profile + "%s%s.txt"%(cut_text,IDstr), unpack=True, delimiter=',')

R_i, P_R_i = np.loadtxt(dist_dir + "distribution_radius_" + profile + "_unperturbed%s.txt"%(IDstr), unpack=True, delimiter=',', usecols=(0,1))
R_f, P_R_f = np.loadtxt(dist_dir + "distribution_radius_" + Rstr + "_" +  profile + "%s%s.txt"%(cut_text,IDstr), unpack=True, delimiter=',', usecols=(0,1))


N_i = np.sum([M_i > 1e-25])
N_f = np.sum([M_f > 1e-25])

p_surv = N_f/N_i

print("Survival probability:", p_surv)
a_plot = 0.7
a_hist = 0.4
xpad=8

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(15,5), gridspec_kw={'wspace': 0.25})
# fig.suptitle(r"$r_\mathrm{GC} = " + Rstr + "\,\mathrm{kpc}$, $p_\mathrm{surv} = %.3f$"%(p_surv,))


# Plot one
# M_bins = np.geomspace(1e-20, 1e-5, 50)
# plt.figure(figsize=(7,5))
# print(M_i.min(), M_f.min())
# Mbins = np.geomspace(M_i.min(), M_i.max(), 50)
# print(M_i.min(), M_i.max())
# logwidth = np.diff(np.log10(Mbins))[0]
# Mbins = np.append(Mbins, Mbins[-1]*(10**logwidth))
# Mbins = np.insert(Mbins, 0, Mbins[0]/(10**logwidth))
# Mbins = np.insert(Mbins, 0, Mbins[0]/(10**logwidth))
# Mbins = np.insert(Mbins, 0, Mbins[0]/(10**logwidth))

cbefore = 'k'
cafter = 'C0'



ax1.plot(M_f[P_M_f>1e-30], M_f[P_M_f>1e-30]*P_M_f[P_M_f>1e-30], color=cafter)
ax1.plot(M_i, M_i*P_M_i, color=cbefore, linestyle='--')


ax1.text(0.55, 0.9, "$r_\\mathrm{GC} = " + Rstr + "\\,\\mathrm{kpc}$\n$p_\\mathrm{surv} = %.2f$"%(p_surv,),transform=ax1.transAxes, ha='left', va='top')
ax1.xaxis.set_tick_params(pad=xpad)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel(r"$M_\mathrm{AMC}$ $[M_\odot]$")
# ax1.set_ylabel(r"$N_\mathrm{AMC}$")
ax1.set_ylabel(r"$M_\mathrm{AMC} \times P(M_\mathrm{AMC})$")
ax1.set_xlim(1e-22, 1e-4)
ax1.set_ylim(1e-12, 10)




# Plot two
# counts, edges, patches = ax2.hist(R_i, bins=np.geomspace(1e-8, 1e0, 50), alpha=a_hist, density=True)
# ax2.step(edges[1:], counts, color='C0', alpha=a_plot)
# counts, edges, patches = ax2.hist(R_f, bins=np.geomspace(1e-8, 1e0, 50), alpha=a_hist, density=True)
# ax2.step(edges[1:], counts, color='C1', alpha=a_plot)
# ax2.plot(R_f_true[P_R_f_true>0.0], P_R_f_true[P_R_f_true>0.0], label='After Disruption', ls='--')
ax2.plot(R_f[P_R_f>1e-30], R_f[P_R_f>1e-30]*P_R_f[P_R_f>1e-30], color=cafter)
ax2.plot(R_i[P_R_i>1e-30], R_i[P_R_i>1e-30]*P_R_i[P_R_i>1e-30], ls='--', color=cbefore)

# ax2.hist(R_i, alpha = a_plot, bins=np.geomspace(1e-9, 1e-3, 50))
# ax2.hist(R_f, alpha = a_plot, bins=np.geomspace(1e-9, 1e-3, 50))

ax2.xaxis.set_tick_params(pad=xpad)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel(r"$R_\mathrm{AMC}$ $[\mathrm{pc}]$")
ax2.set_ylabel(r"$R_\mathrm{AMC} \times P(R_\mathrm{AMC})$")



ax2.set_ylim(1e-15, 10)
ax2.set_xlim(1e-10, 1)
# plt.ylabel("$N_\mathrm{AMC}$")

#ax3.legend(loc='best')
#Rstr_pc = str(int(float(Rstr)*1000))
#plt.savefig("../../plots/Distributions_" + PROFILE + "_r"+Rstr_pc+"pc%s%s.pdf"%(cut_text,IDstr), bbox_inches='tight')

plt.show()