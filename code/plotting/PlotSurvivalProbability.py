import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os


rc('text', usetex=True)
rc('font', size=18)

root_dir = '../../data/'

frac_AScut_NFW = 0.0146239
frac_AScut_PL = 0.0002719

#IDstr = "_ma_400mueV"
#IDstr = "_ma_306mueV"
IDstr = "_ma_41_564mueV_MW_delta_a"
#IDstr = "_wStripping"

#To-Do - make the correction for the initial range of samples masses more concrete

if (IDstr ==  "_gamma-0.5"):
    frac_AScut_NFW = 0.0463458
    frac_AScut_PL = 0.002647
elif (IDstr == "_wStripping"):
    #frac_AScut_NFW = 0.0138*0.9332610321836602
    frac_AScut_NFW = 0.0146239*0.8764842147261401
    frac_AScut_PL = 0.0002719*0.8764842147261401
else:
    frac_AScut_NFW = 0.0146239
    frac_AScut_PL = 0.0002719



AS_CUT = True

cut_text = ""
if (AS_CUT):
    print("> Calculating with axion-star cut...")
    cut_text = "_AScut"
    print("> Initial fraction of AMCs surviving AS cut:")
    print(">    NFW:", frac_AScut_NFW)
    print(">    PL:", frac_AScut_PL)

#------ Plot survival probability (as a function of a) --------------------

alist_PL, psurv_a_PL, psurv_a_AScut_PL = np.loadtxt(root_dir + 'SurvivalProbability_a_PL' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,1,2), unpack=True)
alist_NFW, psurv_a_NFW, psurv_a_AScut_NFW = np.loadtxt(root_dir + 'SurvivalProbability_a_NFW' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,1,2), unpack=True)

plt.figure(figsize=(7,5))
plt.semilogx(alist_PL/1e3, psurv_a_PL, color='C0', linestyle='--')
plt.semilogx(alist_NFW/1e3, psurv_a_NFW, color='C8', linestyle='--')

plt.semilogx(alist_PL/1e3, np.clip(psurv_a_AScut_PL/frac_AScut_PL, 0, 1), color='C0', linestyle="-", label='Power-law')
plt.semilogx(alist_NFW/1e3, np.clip(psurv_a_AScut_NFW/frac_AScut_NFW, 0, 1), color='C8', linestyle="-", label='NFW')

plt.semilogx([1e21, 1e21], 'k-', label = "All AMCs")
plt.semilogx([1e21, 1e21], 'k-.', label = "AS cut")

plt.axvline(x=8.33, color='gray', ls=':', zorder=0)
plt.text(8.33, 0.2, "$r_\odot$", rotation = -90, color='gray')

props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)

plt.xlim(0.1, 50.)
plt.ylim(0,1.1)

plt.yticks(np.linspace(0, 1, 6))

plt.gca().tick_params(axis='x', pad=10)

plt.xlabel('Galactocentric semi-major axis $a$ [kpc]')
plt.ylabel('Survival Probability')
#plt.legend(loc = 'upper left' , fontsize=14)

plt.savefig('../../plots/SurvivalProbability_a' + IDstr + '.pdf', bbox_inches='tight')



#------ Plot survival probability (as a function of R) --------------------


Rlist_PL, psurv_R_PL = np.loadtxt(root_dir + 'SurvivalProbability_R_PL' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
Rlist_NFW, psurv_R_NFW = np.loadtxt(root_dir + 'SurvivalProbability_R_NFW' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
density_NFW_init, density_NFW_final, density_NFW_AScut, density_NFW_AScut_masscut = np.loadtxt(root_dir + 'SurvivalProbability_R_NFW' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(2,4,5,6), unpack=True)
density_PL_init, density_PL_final, density_PL_AScut, density_PL_AScut_masscut = np.loadtxt(root_dir + 'SurvivalProbability_R_PL' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(2,4,5,6), unpack=True)
psurv_R_NFW_masscut = density_NFW_final/density_NFW_init
psurv_R_PL_masscut = density_PL_final/density_PL_init

psurv_R_NFW_AScut = np.clip(density_NFW_AScut/(density_NFW_init*frac_AScut_NFW), 0, 1)
psurv_R_PL_AScut = np.clip(density_PL_AScut/(density_PL_init*frac_AScut_PL), 0, 1)

psurv_R_NFW_AScut_masscut = np.clip(density_NFW_AScut_masscut/(density_NFW_init*frac_AScut_NFW), 0, 1)
psurv_R_PL_AScut_masscut = np.clip(density_PL_AScut_masscut/(density_PL_init*frac_AScut_PL), 0, 1)

Rlist_PL_circ, psurv_R_PL_circ = np.loadtxt(root_dir + 'SurvivalProbability_R_PL_circ' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
Rlist_NFW_circ, psurv_R_NFW_circ = np.loadtxt(root_dir + 'SurvivalProbability_R_NFW_circ' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
#density_NFW_circ_init, density_NFW_circ_final = np.loadtxt(root_dir + 'SurvivalProbability_R_NFW_circ.txt', delimiter =',', dtype='f8', usecols=(2,4), unpack=True)
#psurv_R_NFW_circ_masscut = density_NFW_circ_final/density_NFW_circ_init

plt.figure(figsize=(7,5))
plt.semilogx(Rlist_PL/1.e3, psurv_R_PL, color='C0', linestyle='--')
plt.semilogx(Rlist_PL_circ/1e3, psurv_R_PL_circ, color='C0', linestyle=':')
#plt.semilogx(Rlist_PL_circ/1e3, psurv_R_PL_masscut, color='C0', linestyle='--')
plt.semilogx(Rlist_PL_circ/1e3, psurv_R_PL_AScut, color='C0', linestyle='-', label='Power-law')
#plt.semilogx(Rlist_PL_circ/1e3, psurv_R_PL_AScut_masscut, color='C0', linestyle=':')

plt.semilogx(Rlist_NFW_circ/1e3, psurv_R_NFW_circ, color='C8', linestyle=':')
plt.semilogx(Rlist_NFW/1e3, psurv_R_NFW, color='C8', linestyle='--')
#plt.semilogx(Rlist_NFW_circ/1e3, psurv_R_NFW_masscut, color='C8', linestyle='--')
plt.semilogx(Rlist_PL_circ/1e3, psurv_R_NFW_AScut, color='C8', linestyle='-', label='NFW')
#plt.semilogx(Rlist_NFW_circ/1e3, psurv_R_NFW_AScut_masscut, color='C8', linestyle=':')


print("p_surv at r_Sun (PL):", np.interp(8.3, Rlist_PL_circ/1e3, psurv_R_PL_AScut))
print("p_surv at r_Sun (NFW):", np.interp(8.3, Rlist_PL_circ/1e3, psurv_R_NFW_AScut))


plt.semilogx([1e21, 1e21], 'k:', label = "Circular")
plt.semilogx([1e21, 1e21], 'k--', label = "Eccentric")
#plt.semilogx([1e21, 1e21], 'k--', label = r"Ecc., $M_f > 10\% \,M_i$")
plt.semilogx([1e21, 1e21], 'k-', label = "Ecc.+AS cut")
#plt.semilogx([1e21, 1e21], 'k:', label = "Ecc., AS cut, $M_f > 10\% \,M_i$")

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
plt.legend(loc = 'lower right' , fontsize=12)

plt.savefig('../../plots/SurvivalProbability_R' + IDstr + '.pdf', bbox_inches='tight')


#------ Density as a function of R

Rlist_PL_circ, rho_init_R_PL_circ, rho_R_PL_circ = np.loadtxt(root_dir + 'SurvivalProbability_R_PL_circ' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,2,3), unpack=True)
Rlist_PL, rho_init_R_PL, rho_R_PL = np.loadtxt(root_dir + 'SurvivalProbability_R_PL' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,2,3), unpack=True)
Rlist_NFW, rho_init_R_NFW, rho_R_NFW, rho_R_NFW_masscut, rho_R_NFW_AScut = np.loadtxt(root_dir + 'SurvivalProbability_R_NFW' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,2,3, 4,5), unpack=True)
Rlist_PL, rho_init_R_PL, rho_R_PL, rho_R_PL_masscut, rho_R_PL_AScut = np.loadtxt(root_dir + 'SurvivalProbability_R_PL' + IDstr +'.txt', delimiter =',', dtype='f8', usecols=(0,2,3, 4,5), unpack=True)
#Rlist_NFW_circ, psurv_R_NFW_circ = np.loadtxt(root_dir + 'SurvivalProbability_R_NFW_circ.txt', delimiter =',', dtype='f8', usecols=(0,1), unpack=True)


plt.figure(figsize=(7,5))
plt.loglog(Rlist_PL_circ/1e3, rho_init_R_PL_circ/(4*np.pi*Rlist_PL_circ**2), color='grey', linestyle=':', label="Galactic density profile")
plt.loglog(Rlist_PL/1e3, rho_init_R_PL/(4*np.pi*Rlist_PL**2),color='grey', linestyle='-' , label="Unperturbed AMCs")

plt.loglog(Rlist_PL/1e3, frac_AScut_PL*rho_init_R_PL/(4*np.pi*Rlist_PL**2),'C0', linestyle='-.')
plt.loglog(Rlist_PL/1e3, frac_AScut_NFW*rho_init_R_PL/(4*np.pi*Rlist_PL**2),'C8', linestyle='-.')
#plt.loglog(Rlist_PL/1e3, rho_init_R_PL/(4*np.pi*Rlist_PL**2),'w--' , label=" ")

plt.loglog(Rlist_PL/1e3, rho_R_PL_AScut/(4*np.pi*Rlist_PL**2),color='C0', linestyle='-', label="Power-law")
plt.loglog(Rlist_NFW/1e3, rho_R_NFW_AScut/(4*np.pi*Rlist_NFW**2),color='C8', linestyle='-', label="NFW")

plt.loglog(Rlist_PL/1e3, rho_R_PL/(4*np.pi*Rlist_PL**2),color='C0', linestyle='--')
plt.loglog(Rlist_NFW/1e3, rho_R_NFW/(4*np.pi*Rlist_NFW**2),color='C8', linestyle='--')


plt.loglog(Rlist_PL/1e3, 1e30*rho_R_PL_AScut/(4*np.pi*Rlist_PL**2),color='k', linestyle='--', label=r'Perturbed AMCs')
plt.loglog(Rlist_PL/1e3, 1e30*rho_R_PL_AScut/(4*np.pi*Rlist_PL**2),color='k', linestyle='-.', label=r'Unperturbed AMCs + AS cut')
plt.loglog(Rlist_PL/1e3, 1e30*rho_R_PL_AScut/(4*np.pi*Rlist_PL**2),color='k', linestyle='-', label=r'Perturbed AMCs + AS cut')
#plt.loglog(Rlist_NFW/1e3, 1e30*rho_R_NFW_masscut/(4*np.pi*Rlist_NFW**2),color='k', linestyle='-.', label=r'Perturbed AMCs ($M_f > 10\% M_i$)')


#plt.loglog(Rlist_PL_circ/1e3, rho_R_PL_circ)
#plt.text(8.33, 0.2, "$r_\odot$", rotation = -90, color='gray')

props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)

plt.xlim(0.1, 50.)
plt.ylim(1e-7, 1e2)

plt.xlabel('Galactocentric radius $r$ [kpc]')
plt.ylabel(r'Density of AMCs $\rho(r)$ [$M_\odot$ pc$^{-3}$]')
plt.legend(loc = 'upper right' , fontsize=12)#, ncol=2)

plt.savefig('../../plots/DensityReconstruction' + IDstr + '.pdf', bbox_inches='tight')

#------ Plot encounter rate as a function of R ----------------------------



Rlist_PL, PDF_R_PL = np.loadtxt(root_dir + 'EncounterRate_PL%s%s.txt'%(cut_text,IDstr), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
Rlist_NFW, PDF_R_NFW = np.loadtxt(root_dir + 'EncounterRate_NFW%s%s.txt'%(cut_text,IDstr), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)

Rlist_PL_circ, PDF_R_PL_circ = np.loadtxt(root_dir + 'EncounterRate_PL_circ%s%s.txt'%(cut_text,IDstr), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
Rlist_NFW_circ, PDF_R_NFW_circ = np.loadtxt(root_dir + 'EncounterRate_NFW_circ%s%s.txt'%(cut_text,IDstr), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)

plt.figure(figsize=(7,5))

#factor = 3600*24
factor = 1

plt.loglog(Rlist_PL/1e3, PDF_R_PL*1e3*factor, label='Power-law', color='C0')
plt.semilogx(Rlist_PL_circ/1e3, PDF_R_PL_circ*1e3*factor, color='C0', linestyle='--')

plt.semilogx(Rlist_NFW/1e3, PDF_R_NFW*1e3*factor,  label='NFW', color='C8')
plt.semilogx(Rlist_NFW_circ/1e3, PDF_R_NFW_circ*1e3*factor, color='C8', linestyle='--')



plt.semilogx([1e21, 1e21], 'k-', label = "Eccentric")
plt.semilogx([1e21, 1e21], 'k--', label = "Circular")

plt.axvline(x=8.33, color='gray', ls=':', zorder=0)
plt.text(8.33, 1e-2, "$r_\odot$", rotation = -90, color='gray')

props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)

Gamma_PL = np.trapz(PDF_R_PL, Rlist_PL)*3600*24
Gamma_NFW = np.trapz(PDF_R_NFW, Rlist_NFW)*3600*24

Rlist_PL_up, PDF_R_PL_up = np.loadtxt(root_dir + 'EncounterRate_PL_circ%s_unperturbed%s.txt'%(cut_text,IDstr), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
Gamma_PL_up = np.trapz(PDF_R_PL_up, Rlist_PL_up)*3600*24
Rlist_NFW_up, PDF_R_NFW_up = np.loadtxt(root_dir + 'EncounterRate_NFW_circ%s_unperturbed%s.txt'%(cut_text,IDstr), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
Gamma_NFW_up = np.trapz(PDF_R_NFW_up, Rlist_NFW_up)*3600*24

print("Gamma_PL:", Gamma_PL)
print("Gamma_PL (unperturbed):", Gamma_PL_up)
print("Gamma_NFW:", Gamma_NFW)
print("Gamma_NFW (unperturbed):", Gamma_NFW_up)

#if (AS_CUT):
#    plt.title("Including axion star cut")

plt.text(0.35, 0.9, r"$\Gamma_\mathrm{NFW} = %.1f\,\,\mathrm{day}^{-1}$"%(Gamma_NFW),bbox=props,transform=plt.gca().transAxes, fontsize=16, color='C8')
plt.text(0.35, 0.82, r"$\Gamma_\mathrm{PL} = %.1f\,\,\mathrm{day}^{-1}$"%(Gamma_PL),bbox=props, transform=plt.gca().transAxes, fontsize=16, color='C0')

plt.xlim(0.1, 50.)
#plt.ylim(1e-2,1e3)
plt.yticks(np.geomspace(1e-7, 1, 8))
plt.ylim(1e-7, 1)

plt.xlabel('Galactocentric radius $r$ [kpc]')
plt.ylabel('Encounter rate $\mathrm{d}\Gamma/\mathrm{d}r$ [kpc$^{-1}$ s$^{-1}$]')
plt.legend(loc = 'upper left' , fontsize=15)

plt.savefig('../../plots/EncounterRate%s%s.pdf'%(cut_text,IDstr), bbox_inches='tight')

#------Plot also with linear x-scale
plt.xscale('linear')
plt.xlim(0, 30)
plt.savefig('../../plots/EncounterRate_linear%s%s.pdf'%(cut_text,IDstr), bbox_inches='tight')


plt.show()
