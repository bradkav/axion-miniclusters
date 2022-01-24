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
args = parser.parse_args()
profile = args.profile

AS_CUT = False

#Code for procedurally escaping latex characters (such as underscores...)
def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)



#------ Plot survival probability (as a function of a) --------------------

alist, psurv_a, psurv_a_AScut = np.loadtxt(root_dir + 'SurvivalProbability_a_' + profile +'.txt', delimiter =',', dtype='f8', usecols=(0,1,2), unpack=True)

plt.figure(figsize=(7,5))
plt.semilogx(alist/1e3, psurv_a, color='C0', linestyle='-')


plt.axvline(x=8.33, color='gray', ls=':', zorder=0)
plt.text(8.33, 0.2, "$r_\odot$", rotation = -90, color='gray')

props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)

plt.xlim(0.1, 50.)
plt.ylim(0,1.1)

plt.yticks(np.linspace(0, 1, 6))

plt.gca().tick_params(axis='x', pad=10)

plt.xlabel('Galactocentric semi-major axis $a$ [kpc]')
plt.ylabel('Survival Probability')
plt.title(tex_escape(profile))
#plt.legend(loc = 'upper left' , fontsize=14)

#plt.savefig('../../plots/SurvivalProbability_a' + IDstr + '.pdf', bbox_inches='tight')



#------ Plot survival probability (as a function of R) --------------------


Rlist, psurv_R = np.loadtxt(root_dir + 'SurvivalProbability_R_' + profile +'.txt', delimiter =',', dtype='f8', usecols=(0,1), unpack=True)


#Rlist_circ, psurv_R_circ = np.loadtxt(root_dir + 'SurvivalProbability_R_circ' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
plt.figure(figsize=(7,5))
plt.semilogx(Rlist/1.e3, psurv_R, color='C0', linestyle='-')

print("p_surv at r_Sun:", np.interp(8.3, Rlist/1e3, psurv_R))


plt.axvline(x=8.33, color='gray', ls=':', zorder=0)
plt.text(8.33, 0.65, "$r_\odot$", rotation = -90, color='gray')
#plt.text(2.1, 0.5, r"NFW, ecc., $M_f > 10\% \,M_i$", rotation = 65, color='C8', fontsize=12, ha='center', va='center')

props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)

plt.xlim(0.1, 50.)
plt.ylim(0,1.1)

plt.yticks(np.linspace(0, 1, 6))
plt.gca().tick_params(axis='x', pad=10)

plt.xlabel('Galactocentric radius $r$ [kpc]')
plt.ylabel('Survival Probability')
#plt.legend(loc = 'lower right' , fontsize=12)
plt.title(tex_escape(profile))

#plt.savefig('../../plots/SurvivalProbability_R' + IDstr + '.pdf', bbox_inches='tight')


#------ Plot encounter rate as a function of R ----------------------------


Rlist, PDF_R = np.loadtxt(root_dir + 'EncounterRate_%s.txt'%(profile), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)

plt.figure(figsize=(7,5))

plt.loglog(Rlist/1e3, PDF_R*1e3)

#plt.semilogx([1e21, 1e21], 'k-', label = "Eccentric")
#plt.semilogx([1e21, 1e21], 'k--', label = "Circular")

plt.axvline(x=8.33, color='gray', ls=':', zorder=0)
plt.text(8.33, 1e-2, "$r_\odot$", rotation = -90, color='gray')

props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)

Gamma = np.trapz(PDF_R, Rlist)*3600*24

print("Encounter rate (1/day):", Gamma)


plt.text(0.35, 0.9, r"$\Gamma = %.1f\,\,\mathrm{day}^{-1}$"%(Gamma),bbox=props,transform=plt.gca().transAxes, fontsize=16, color='C0')

plt.xlim(0.1, 50.)
#plt.ylim(1e-2,1e3)
#plt.yticks(np.geomspace(1e-7, 1, 8))
#plt.ylim(1e-7, 1)

plt.xlabel('Galactocentric radius $r$ [kpc]')
plt.ylabel('Encounter rate $\mathrm{d}\Gamma/\mathrm{d}r$ [kpc$^{-1}$ s$^{-1}$]')
#plt.legend(loc = 'upper left' , fontsize=15)

plt.title(tex_escape(profile))

#plt.savefig('../../plots/EncounterRate%s%s.pdf'%(cut_text,IDstr), bbox_inches='tight')



plt.show()