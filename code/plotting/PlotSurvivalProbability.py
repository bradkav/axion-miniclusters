import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

import sys
sys.path.append("../")

import tools
import dirs

import re
import argparse

rc('text', usetex=True)
rc('font', size=18)

root_dir = dirs.data_dir

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
def plot_psurv_a(profile, mass_function_ID, IDstr="", save_plot=False, show_plot=True):
    
    file_suffix = tools.generate_suffix(profile, mass_function_ID, circular=False,
                                        AScut=False, unperturbed=False, IDstr=IDstr)


    alist, psurv_a, psurv_a_AScut = np.loadtxt(root_dir + 'SurvivalProbability_a_' + file_suffix + '.txt', delimiter =',', dtype='f8', usecols=(0,1,2), unpack=True)

    plt.figure(figsize=(7,5))
    plt.semilogx(alist, psurv_a, color='C0', linestyle='-', label="All")
    plt.semilogx(alist, psurv_a_AScut, color='C0', linestyle='--', label="w/ AS cut")

    if (INCLUDE_SOLAR_RADIUS):
        plt.axvline(x=8.33, color='gray', ls=':', zorder=0)
        plt.text(8.33, 0.2, r"$r_\odot$", rotation = -90, color='gray')

    props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)

    #plt.xlim(0.1, 50.)
    plt.ylim(0,1.1)

    plt.yticks(np.linspace(0, 1, 6))

    plt.gca().tick_params(axis='x', pad=10)

    plt.xlabel('Galactocentric semi-major axis $a$ [pc]')
    plt.ylabel('Survival Probability')
    plt.title(f"Density profile: {profile}; Mass function: {mass_function_ID.replace('_', '-')}")
    
    plt.legend(loc='upper left', fontsize=14)
    
    if (save_plot):
        plt.savefig(dirs.plot_dir + 'SurvivalProbability_a_' + file_suffix + '.pdf', bbox_inches='tight')
    
    if (show_plot): 
        plt.show()

#------ Plot survival probability (as a function of R) --------------------
def plot_psurv_r(profile, mass_function_ID,  IDstr="", circular=False, save_plot=False, show_plot=True):

    file_suffix = tools.generate_suffix(profile, mass_function_ID, circular=circular,
                                        AScut=False, unperturbed=False, IDstr=IDstr)

    Rlist, psurv_R, psurv_R_AScut = np.loadtxt(root_dir + 'SurvivalProbability_R_' + file_suffix +'.txt', delimiter =',', dtype='f8', usecols=(0, 1, 2), unpack=True)

    plt.figure(figsize=(7,5))
    plt.semilogx(Rlist, psurv_R, color='k', linestyle='-', label="All")
    plt.semilogx(Rlist, psurv_R_AScut, color='k', linestyle='--', label="w/ AS cut")
    #plt.semilogx(Rlist/1.e3, psurv_R_AScut, color='k', linestyle='-')

    print("p_surv at r_Sun:", np.interp(8.3, Rlist/1e3, psurv_R))

    if (INCLUDE_SOLAR_RADIUS):
        plt.axvline(x=8.33e3, color='gray', ls=':', zorder=0)
        plt.text(8.33e3, 0.65, r"$r_\odot$", rotation = -90, color='gray')
    #plt.text(2.1, 0.5, r"NFW, ecc., $M_f > 10\% \,M_i$", rotation = 65, color='C8', fontsize=12, ha='center', va='center')

    props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)


    plt.ylim(0,1.1)

    plt.yticks(np.linspace(0, 1, 6))
    plt.gca().tick_params(axis='x', pad=10)

    plt.xlabel('Galactocentric radius $r$ [pc]')
    plt.ylabel('Survival Probability')
    #plt.legend(loc = 'lower right' , fontsize=12)
    plt.title(f"Density profile: {profile}; Mass function: {mass_function_ID.replace('_', '-')}")
    
    plt.legend(loc='upper left', fontsize=14)
    
    if (save_plot):
        plt.savefig(dirs.plot_dir + 'SurvivalProbability_R_' + file_suffix + '.pdf', bbox_inches='tight')

    if (show_plot):
        plt.show()
        
#-------- Plot encounter rate distribution ----------

def plot_encounter_rate(profile, mass_function_ID,  IDstr="", circular=False, save_plot=False, show_plot=True):

    file_suffix = tools.generate_suffix(profile, mass_function_ID, circular=False,
                                        AScut=False, unperturbed=False, IDstr=IDstr)

    plt.figure(figsize=(7,5))

    #plt.semilogx([1e21, 1e21], 'k-', label = "Eccentric")
    #plt.semilogx([1e21, 1e21], 'k--', label = "Circular")
    if (INCLUDE_SOLAR_RADIUS):
        plt.axvline(x=8.33*1e3, color='gray', ls=':', zorder=0)
        plt.text(8.33*1e3, 1e-2, r"$r_\odot$", rotation = -90, color='gray')

    props = dict(boxstyle='round', facecolor='white',edgecolor='white', alpha=0.9)


    Rlist, PDF_R, PDF_R_AScut = np.loadtxt(root_dir + 'EncounterRate_' + file_suffix + '.txt', delimiter =',', dtype='f8', usecols=(0,1,2), unpack=True)

    
    plt.loglog(Rlist, PDF_R, color='k', label="All")
    plt.loglog(Rlist, PDF_R_AScut, color='k', linestyle='--', label="w/ AS cut")

    Gamma = np.trapz(PDF_R, Rlist)*3600*24
    Gamma_AScut = np.trapz(PDF_R_AScut, Rlist)*3600*24

    print("Encounter rate (without AScut)[day^-1]:\t", Gamma)
    print("Encounter rate (including AScut) [day^-1]:\t", Gamma_AScut)


    plt.text(0.35, 0.9, r"$\Gamma (\textrm{all}) = %.1f\,\,\mathrm{day}^{-1}$"%(Gamma),bbox=props,transform=plt.gca().transAxes, fontsize=16, color='k')

    #plt.xlim(0.1, 50.)
    #plt.ylim(1e-2,1e3)
    #plt.yticks(np.geomspace(1e-7, 1, 8))
    plt.ylim(1e-10, 1e-3)

    plt.xlabel(r'Galactocentric radius $r$ [pc]')
    plt.ylabel(r'Encounter rate $\mathrm{d}\Gamma/\mathrm{d}r$ [pc$^{-1}$ s$^{-1}$]')
    #plt.legend(loc = 'upper left' , fontsize=15)
    plt.title(f"Density profile: {profile}; Mass function: {mass_function_ID.replace('_', '-')}")

    plt.legend(loc='upper left', fontsize=14)

    if (save_plot):
        plt.savefig(dirs.plot_dir + 'EncounterRate' + file_suffix + '.pdf', bbox_inches='tight')

    if (show_plot):
        plt.show()

