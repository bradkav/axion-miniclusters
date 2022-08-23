import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

import re
import argparse

rc('text', usetex=True)
rc('font', size=18)

root_dir = '../../data/'

parser = argparse.ArgumentParser(description='...')
parser.add_argument('-profile','--profile', help='Density profile for AMCs - `NFW` or `PL`', type=str, default="PL")

parser.add_argument(
    "-AScut",
    "--AScut",
    dest="AScut",
    action="store_true",
    help="Include an axion star cut on the AMC properties.",
)

args = parser.parse_args()
profile = args.profile

IDstr = "_ma_8_270mueV_M31_delta_p"

AS_CUT = args.AScut
cut_text = ""
if (AS_CUT):
    cut_text = "_AScut"


INCLUDE_SOLAR_RADIUS = False

#------ Plot survival probability (as a function of a) --------------------

alist, psurv_a, psurv_a_AScut = np.loadtxt(root_dir + 'SurvivalProbability_a_' + profile + IDstr + '.txt', delimiter =',', dtype='f8', usecols=(0,1,2), unpack=True)

plt.figure(figsize=(7,5))
plt.semilogx(alist/1e3, psurv_a, color='C0', linestyle='-')

if (INCLUDE_SOLAR_RADIUS):
    plt.axvline(x=8.33, color='gray', ls=':', zorder=0)
    plt.text(8.33, 0.2, "$r_\odot$", rotation = -90, color='gray')

props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)

#plt.xlim(0.1, 50.)
plt.ylim(0,1.1)

plt.yticks(np.linspace(0, 1, 6))

plt.gca().tick_params(axis='x', pad=10)

plt.xlabel('Galactocentric semi-major axis $a$ [kpc]')
plt.ylabel('Survival Probability')
plt.title(profile)
#plt.legend(loc = 'upper left' , fontsize=14)

#plt.savefig('../../plots/SurvivalProbability_a' + IDstr + '.pdf', bbox_inches='tight')



#------ Plot survival probability (as a function of R) --------------------


Rlist, psurv_R, psurv_R_AScut = np.loadtxt(root_dir + 'SurvivalProbability_R_' + profile + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0, 1, 5), unpack=True)


#Rlist_circ, psurv_R_circ = np.loadtxt(root_dir + 'SurvivalProbability_R_circ' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
plt.figure(figsize=(7,5))
plt.semilogx(Rlist, psurv_R, color='k', linestyle='-')
plt.semilogx(Rlist, psurv_R_AScut, color='k', linestyle='-')
#plt.semilogx(Rlist/1.e3, psurv_R_AScut, color='k', linestyle='-')

print("p_surv at r_Sun:", np.interp(8.3, Rlist/1e3, psurv_R))

if (INCLUDE_SOLAR_RADIUS):
    plt.axvline(x=8.33e3, color='gray', ls=':', zorder=0)
    plt.text(8.33e3, 0.65, "$r_\odot$", rotation = -90, color='gray')
#plt.text(2.1, 0.5, r"NFW, ecc., $M_f > 10\% \,M_i$", rotation = 65, color='C8', fontsize=12, ha='center', va='center')

props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)

#plt.xlim(0.1, 50.)
plt.ylim(0,1.1)

plt.yticks(np.linspace(0, 1, 6))
plt.gca().tick_params(axis='x', pad=10)

plt.xlabel('Galactocentric radius $r$ [pc]')
plt.ylabel('Survival Probability')
#plt.legend(loc = 'lower right' , fontsize=12)
plt.title(profile)

plt.savefig('../../plots/SurvivalProbability_R_' + profile + IDstr + '.pdf', bbox_inches='tight')


#------ Plot encounter rate as a function of R ----------------------------




plt.figure(figsize=(7,5))



#plt.semilogx([1e21, 1e21], 'k-', label = "Eccentric")
#plt.semilogx([1e21, 1e21], 'k--', label = "Circular")
if (INCLUDE_SOLAR_RADIUS):
    plt.axvline(x=8.33*1e3, color='gray', ls=':', zorder=0)
    plt.text(8.33*1e3, 1e-2, "$r_\odot$", rotation = -90, color='gray')

props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)


for i in [3,]:
    IDstr2 = IDstr
    if (i == 4):
        IDstr2 += "_old"
    
    Rlist, PDF_R = np.loadtxt(root_dir + 'EncounterRate_%s%s%s.txt'%(profile, cut_text, IDstr2), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)

    plt.loglog(Rlist, PDF_R*1e3, color='C' + str(i))

    Gamma = np.trapz(PDF_R, Rlist)*3600*24

    print("Encounter rate (1/day):", Gamma)

    ytext = 0.8
    if (i == 4):
        ytext = 0.9
    plt.text(0.35, ytext, r"$\Gamma = %.1f\,\,\mathrm{day}^{-1}$"%(Gamma),bbox=props,transform=plt.gca().transAxes, fontsize=16, color='C' + str(i))

#plt.xlim(0.1, 50.)
#plt.ylim(1e-2,1e3)
#plt.yticks(np.geomspace(1e-7, 1, 8))
plt.ylim(1e-7, 1)

plt.xlabel('Galactocentric radius $r$ [pc]')
plt.ylabel('Encounter rate $\mathrm{d}\Gamma/\mathrm{d}r$ [kpc$^{-1}$ s$^{-1}$]')
#plt.legend(loc = 'upper left' , fontsize=15)

plt.title(profile)

plt.savefig('../../plots/EncounterRate%s%s_comparison.pdf'%(cut_text,IDstr), bbox_inches='tight')



plt.show()
