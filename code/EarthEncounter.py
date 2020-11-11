import numpy as np
import matplotlib.pyplot as plt
import NSencounter as NE
import perturbations as PB
from scipy.integrate import quad
from scipy import interpolate
import mass_function

fAMC    = 1                # fraction of AMCs as DM
speedc  = 9.721e-9         # pc/s
Rsun    = 8.33e3           # pc
MsunGeV = 1.11580328e57    # Mass of the Sun in GeV
pc      = 3.086e18         # cm
year    = 3.15e7           # s
rho_loc = NE.rhoNFW(Rsun)  # local DM energy density
speedu = np.sqrt(8/np.pi)*(2.9/3.0)*1.e-3*speedc

### 
# We set the axion mass m_a=20\mueV
# This mass corresponds roughly to an axion decay
# constant of 3e11 and a confinement scale of Lambda = 0.076
###
in_maeV   = 2.e-5   # axion mass in eV
in_gg     = -0.7     # This leads to the HMF \propto M^-0.7

AMC_MF = mass_function.PowerLawMassFunction(m_a = in_maeV, gamma = in_gg)

pref = fAMC*rho_loc/AMC_MF.mavg*speedu # n*u in cm^-2 s^-1

## Define the location of the files containing the radial and mass profiles of the AMCs
MCdata_path = "../data/distributions/"
fM_PL   = MCdata_path + 'distribution_mass_8.63_PL_AScut.txt'
fM_NFW  = MCdata_path + 'distribution_mass_8.63_NFW_AScut.txt'
fr_PL   = MCdata_path + 'distribution_radius_PL_circ_AScut_unperturbed.txt'
fr_NFW  = MCdata_path + 'distribution_radius_NFW_circ_AScut_unperturbed.txt'
fr_PLc  = MCdata_path + 'distribution_radius_8.63_PL_AScut.txt'
fr_NFWc = MCdata_path + 'distribution_radius_8.63_NFW_AScut.txt'

## AS cuts
#BJK: Double-check these numbers?
#cut_PL    = 0.000268627187253961087
#cut_NFW   = 0.00762279608818846192

cut_PL = 2.7e-4
cut_NFW = 1.5e-2

## Survival probabilities
psurv_PL  = 0.99
psurv_NFW = 0.46

## Read the mass profile of the AMCs at Earth orbit and find the avg mass
M_PL,   probM_PL   = np.loadtxt(fM_PL,   delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)
M_NFW,  probM_NFW  = np.loadtxt(fM_NFW,  delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)
Mavg_PL   = np.trapz(probM_PL*M_PL,  M_PL)
Mavg_NFW  = np.trapz(probM_NFW*M_NFW,M_NFW)

## Read the radial profile of the AMCs at Earth orbit
r_PL,   probr_PL   = np.loadtxt(fr_PL,   delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)
r_NFW,  probr_NFW  = np.loadtxt(fr_NFW,  delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)
r_PLc,  probr_PLc  = np.loadtxt(fr_PLc,  delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)
r_NFWc, probr_NFWc = np.loadtxt(fr_NFWc, delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)

# Mean R^2 of the AMCs
R2_PL   = np.trapz(probr_PL  * np.pi* r_PL**2,  r_PL)
R2_NFW  = np.trapz(probr_NFW * np.pi* r_NFW**2, r_NFW)
R2_PLc  = np.trapz(probr_PLc * np.pi* r_PLc**2, r_PLc)
R2_NFWc = np.trapz(probr_NFWc* np.pi* r_NFWc**2,r_NFWc)

# Encounter rate for unpertubed AMCs
Gamma_PL   = cut_PL*  pref* R2_PL
Gamma_NFW  = cut_NFW* pref* R2_NFW

# Encounter rate for pertubed AMCs
Gamma_PLc  = psurv_PL*  cut_PL*  pref* R2_PLc
Gamma_NFWc = psurv_NFW* cut_NFW* pref* R2_NFWc

time_PL   = 1./Gamma_PL/year
time_PLc  = 1./Gamma_PLc/year
time_NFW  = 1./Gamma_NFW/year
time_NFWc = 1./Gamma_NFWc/year

print("Encounter rate for unperturbed AMCs [year^-1]")
print("    PL: %.3e"%(time_PL,))
print("    NFW: %.3e"%(time_NFW,))
print(" ")
print("Encounter rate for perturbed AMCs [year^-1]")
print("    PL: %.3e"%(time_PLc,))
print("    NFW: %.3e"%(time_NFWc,))
