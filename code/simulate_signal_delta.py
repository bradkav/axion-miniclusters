#!/usr/bin/env python3
import numpy as np

from scipy import interpolate

import NSencounter as NE
import perturbations as PB

from tqdm import tqdm
import argparse

import os

import mass_function

import params

#print(NE.__file__)

###
###  This code returns the simulated signal from axion MC - NS encounter
###

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_size = comm.Get_size()
    MPI_rank = comm.Get_rank()
    if (MPI_size > 1): USING_MPI = True
    
except ImportError as err:
    print("   mpi4py module not found: using a single process only...")
    USING_MPI = False
    MPI_size = 1
    MPI_rank = 0

print(MPI_size, MPI_rank)

import dirs
if not os.path.exists(dirs.data_dir + "distributions/"):
    os.makedirs(dirs.data_dir + "distributions/")


# constants
hbar   = 6.582e-16      # GeV/GHz
GaussToGeV2 = 1.953e-20 # GeV^2
Tage   = 4.26e17
RSun   = 8.33e3      # pc
pc     = 3.086e16    # pc in m
cs     = 3.e8        # speed of light in m/s
vrel0  = 1.e-3
vrel   = vrel0*cs/pc # relative velocity in pc/s
u_dispersion = 1.e-11 # velocity dispersion in pc/s


ma     = params.m_a*1e-9   # axion mass in GeV
maHz   = ma/hbar        # axion mass in GHz


bandwidth0 = vrel0**2/(2.*np.pi)*maHz*1.e9 # Bandwidth in Hz
maeV   = ma*1.e9        # axion mass in eV
min_enhancement = params.min_enhancement    #Threshold for 'interesting' encounters (k = 10% of the local DM density)

f_AMC = 1.0 #Fraction of Dark Matter in the form of axion miniclusters

M_cut  = 1.0e-29

#This mass corresponds roughly to an axion decay 
#constant of 3e11 and a confinement scale of Lambda = 0.076
in_maeV   = maeV        # axion mass in eV
in_gg     = -0.7        

#IDstr = "_ma_306mueV"
IDstr = params.IDstr


print("> Using m_a = %.2e eV, gamma = %.2f"%(in_maeV, in_gg))




## Neutron Star characteristics
MNS = 1.4     # MSun
RNS = 3.24e-13 # pc -- This corresponds to 10km
Pm = -0.3*np.log(10)  # Period peak
Ps = 0.34 # Spread of period
Bm = 12.65*np.log(10) # B field Peak peak
Bs = 0.55 # B field Peak spread

#Define properties of the AS cut
def r_AS(M_AMC):
    m_22 = in_maeV/1e-22
    return 1e3*(1.6/m_22)*(M_AMC/1e9)**(-1/3)

alpha_AS = r_AS(1.0)
k_AMC = (3/(4*np.pi))**(1/3)

#---------------------------------
#-------- OPTIONS ----------------
#---------------------------------


parser = argparse.ArgumentParser(description='...')

parser.add_argument('-profile','--profile', help='Density profile for AMCs - `NFW` or `PL`', type=str, default="PL")
parser.add_argument('-unperturbed', '--unperturbed', help='Calculate for unperturbed profiles?', type=int, default=0)
parser.add_argument("-AScut", "--AScut", dest="AScut", action = 'store_true', help="Include an axion star cut on the AMC properties.")
parser.add_argument("-circ", "--circ", dest="circ", action = 'store_true', help="Enforce circular orbits.")
parser.add_argument('-Ne', '--Ne', help="Number of signal events to simulate.", type=float, default=1e7)
parser.add_argument("-mass_choice", "--mass_choice", help = "Mass parameter = 'c' or 'a' for characteristic or average.", type = str, default="c")
parser.set_defaults(AScut=False)
parser.set_defaults(circ=False)

args = parser.parse_args()
if (args.unperturbed <= 0):
    UNPERTURBED = False
else: 
    UNPERTURBED = True
                    
Ne = int(args.Ne)
    
PROFILE = args.profile

AS_CUT = args.AScut
cut_text = ""
if (AS_CUT):
    print("> Calculating with axion-star cut...")
    cut_text = "_AScut"

#Mass function
if (PROFILE == "PL" or UNPERTURBED == True):
    AMC_MF = mass_function.PowerLawMassFunction(m_a = in_maeV, gamma = in_gg)
elif (PROFILE == "NFW"):
    AMC_MF = mass_function.StrippedPowerLawMassFunction(m_a = in_maeV, gamma = in_gg)

CIRC = args.circ
circ_text = ""
if (CIRC):
    circ_text = "_circ"

MASS_CHOICE = args.mass_choice

IDstr += "_delta_" + MASS_CHOICE.lower()

                    
plt_path = "../plots/"
dist_path = dirs.data_dir+"distributions/"

#Load in survival probabilities
#a_surv_file = abs_path+"SurvivalProbability_a_" + PROFILE + ".txt" #List of survival probabilities
R_surv_file = dirs.data_dir+"SurvivalProbability_R_" + PROFILE + circ_text + IDstr + ".txt" #List of survival probabilities
#a_list, psurv_a_list = np.loadtxt(a_surv_file, delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
R_list, psurv_R_list = np.loadtxt(R_surv_file, delimiter =',', dtype='f8', usecols=(0,1), unpack=True)


#Load in encounter rates
if (UNPERTURBED):
    encounter_file =  dirs.data_dir+"EncounterRate_" + PROFILE + "_circ%s_unperturbed%s.txt"%(cut_text,IDstr) #List of encounter rates
else:
    encounter_file =  dirs.data_dir+"EncounterRate_" + PROFILE + "%s%s%s.txt"%(circ_text,cut_text,IDstr) #List of encounter rates
R_list, dGammadR_list = np.loadtxt(encounter_file, delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
dGammadR_list *= f_AMC

#Generate some interpolation functions
#psurv_a     = interpolate.interp1d(a_list, psurv_a_list) # survival probability (as a function of a)
#psurv_R     = interpolate.interp1d(R_list, psurv_R_list) # survival probability (as a function of R)
dGammadR    = interpolate.interp1d(R_list, dGammadR_list)  # PDF of the galactic radius


dist_r_list = []
dist_rho_list = []

dict_interp_r = dict()
dict_interp_r_corr = dict()
dict_interp_rho = dict()
dict_interp_z   = dict()
dict_interp_mass = dict()

# --------------------- First we prepare the sampling distributions and total interactions

if (UNPERTURBED):
    dist_r, dist_Pr, dist_Pr_sigu  = np.loadtxt(dist_path + 'distribution_radius_%s_circ%s_unperturbed%s.txt'%(PROFILE,cut_text, IDstr), delimiter =', ', dtype='f8', usecols=(0,1,2), unpack=True)
    #dist_rho, dist_P_rho = np.loadtxt(dist_path + 'distribution_rho_%s_unperturbed.txt'%(PROFILE,), delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)
    interp_r = interpolate.interp1d(dist_r, dist_Pr)
    interp_r_corr = interpolate.interp1d(dist_r, dist_Pr_sigu)
    #interp_rho = interpolate.interp1d(dist_rho[dist_P_rho>0.0], dist_P_rho[dist_P_rho>0.0], bounds_error=False, fill_value=(0, 0)) # Density distribution dPdrho

else:

    # Loop through distances
    for i, R in enumerate(R_list):
        R_kpc = R/1e3

    
        # distRX is AMC radius
        # distRY is the PDF dP/dR (normalised as int dP/dR dR = 1) where R is the AMC radius
        # distRC is the PDF <sigma u>*dP/dR where R is the AMC radius
        # distDX is the list of densities for dPdrho
        # distDY is the dPdrho
        
        
        try:
            distRX, distRY, distRC = np.loadtxt(dist_path + 'distribution_radius_%.2f_%s%s%s%s.txt'%(R_kpc, PROFILE, circ_text, cut_text, IDstr), delimiter =', ', dtype='f8', usecols=(0,1,2), unpack=True)
            #distDX, distDY = np.loadtxt(dist_path + 'distribution_rho_%.2f_%s.txt'%(R_kpc, PROFILE), delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)
    
            dist_r_list.append(distRX)
            dict_interp_r[i] = interpolate.interp1d(distRX, distRY) # Radius distribution
            dict_interp_r_corr[i] = interpolate.interp1d(distRX, distRC) # Corrected radius distr 
            distMX, distMY = np.loadtxt(dist_path + 'distribution_mass_%.2f_%s%s%s%s.txt'%(R_kpc, PROFILE, circ_text, cut_text, IDstr), delimiter =', ', unpack=True)
            dict_interp_mass[i] = interpolate.interp1d(distMX, distMY, bounds_error=False, fill_value=0.0)
        
        except:
            print("Warning: File for radius %.2f does not exist" % R_kpc)
            dist_r_list.append(None)
            dist_rho_list.append(None)
            dict_interp_r[i] = None
            dict_interp_r_corr[i] = None
            #dict_interp_rho[i] = None
            
            #if (PROFILE == "NFW"):
            dict_interp_mass[i] = None
            
            continue

GammaTot = np.trapz(dGammadR_list, R_list) # s^-1
print("Gamma = ", GammaTot, "s^-1 [for f_AMC = 1]")



# --------------------- Below we are computing the Signal

#BJK: This still needs lots of cleaning up! A lot of this may be left-over useless stuff.

Interactions = []
Interactions_SJW = []

Prd   = np.random.lognormal(Pm, Ps, Ne) # Period of the NS in s^-1
Bfld  = np.random.lognormal(Bm, Bs, Ne) # B field of NS in gauss
theta = np.random.uniform(-np.pi/2.,np.pi/2., Ne)   # B field orientation FIXME: misalignment angle?

R_sample = NE.inverse_transform_sampling_log(dGammadR,  [np.min(R_list[psurv_R_list > 2e-6]), np.max(R_list)],  n_samples=Ne) # Galactocentric radius of interaction in pc
R_sample = np.array(R_sample)

# Draw the radii and densities of the axion MCs
Z_gal = np.zeros(Ne)
MC_r = np.zeros(Ne)
MC_rho = np.zeros(Ne)


for l, R in enumerate(tqdm(R_sample)):
    
    # Draw a value of z for the encounter
    dpdz  = lambda Z: NE.dPdZ(R, Z) 
    Z_gal[l] = NE.inverse_transform_sampling(dpdz, np.array([-R,R]), n_samples=1)

    if (not UNPERTURBED):
        # Find the GC radius that is closer to what is drawn from distribution
        abs_val = np.abs(R_list - R)
        smallest = abs_val.argmin()
        R_orb   = R_list[smallest] # Closest orbit to the r drawn
        dist_r  = dist_r_list[smallest] # List of MC radii
        #dist_rho  = dist_rho_list[smallest] # List of MC densities
        #interp_rho = dict_interp_rho[smallest]
        interp_r_corr = dict_interp_r_corr[smallest]
        
        #if (PROFILE == "NFW"):
        interp_M = dict_interp_mass[smallest]

        
    # radius in pc
    MC_r[l] = NE.inverse_transform_sampling_log(interp_r_corr, dist_r, n_samples=1, nbins=1000)
    

    rho_max = 3*1e1*AMC_MF.mmax/(4*np.pi*MC_r[l]**3)
    if (AS_CUT):
        rho_min_AS = (alpha_AS*k_AMC/MC_r[l]**2)**3
    else:
        rho_min_AS = 1e-30

    #Sample AMC density rho given the AMC radius R
    if (UNPERTURBED):
        
        dPdM = lambda x: AMC_MF.dPdlogM(x)/x
        #(M_f/rho)*P(M_f)
        P_rho_given_r = lambda rho: ((4*np.pi/3)*MC_r[l]**3)*dPdM((4*np.pi/3)*rho*MC_r[l]**3)
        rho_min = np.maximum(3*AMC_MF.mmin/(4*np.pi*MC_r[l]**3), rho_min_AS)
        
    else:
        P_rho_given_r = lambda rho: ((4*np.pi/3)*MC_r[l]**3)*interp_M((4*np.pi/3)*rho*MC_r[l]**3)
        rho_min = np.maximum(3*(1e-6*AMC_MF.mmin)/(4*np.pi*MC_r[l]**3), rho_min_AS)
    #P_rho_given_r = lambda rho: NE.P_r_given_rho(MCrad[l], rho, mmin, mmax, gg)*dict_interp_rho[smallst](rho)/dict_interp_rad[smallst](MCrad[l])
    
    MC_rho[l] = NE.inverse_transform_sampling_log(P_rho_given_r, [rho_min, rho_max], n_samples=1, nbins=1000)
    #print(rho_min, MC_rho[l])
    

psi   = np.random.uniform(-np.pi,np.pi,Ne)  # Longitude
xi    = np.arcsin(Z_gal/R_sample)           # Latitude


z0    = Z_gal
x0    = R_sample*np.cos(xi)*np.cos(psi)
y0    = R_sample*np.cos(xi)*np.sin(psi)

s0    = np.sqrt((RSun + x0)**2 + y0**2 + z0**2) # Distance from events in pc
bG    = np.arctan(z0/np.sqrt((RSun + x0)**2 + y0**2))*180.0/np.pi # Galactic Latitude
lG    = np.arctan2(y0,(RSun + x0))*180.0/np.pi # Galactic Longitude

#Relative velocity between NS & AMC
vrel = np.sqrt(2)*PB.sigma(R_sample)*3.24078e-14
ux    = np.random.normal(0, vrel, Ne) # pc/s
uy    = np.random.normal(0, vrel, Ne) # pc/s
uz    = np.random.normal(0, vrel, Ne) # pc/s
#ut    = np.abs(np.sqrt((ux + VNSX)**2 + (uy + VNSY)**2 + uz**2)) # pc/s
ut    = np.sqrt(ux**2 + uy**2 + uz**2) # pc/s

# BJK: Here, we need to make sure that we implement the same cut on the
# impact parameter as we did for the radii of very diffuse AMCs
rho_loc = NE.rhoNFW(R_sample)
rho_crit = rho_loc*min_enhancement


if (PROFILE == "PL"):
    r_cut = MC_r*np.minimum((MC_rho/(4*rho_crit))**(4/9), np.ones_like(MC_rho))
    
elif (PROFILE == "NFW"):
    c = 100
    rho_s = MC_rho*(c**3/(3*NE.f_NFW(c))) #Convert mean density rhoi to AMC density
    r_cut = MC_r*np.minimum(NE.x_of_rho(rho_crit/rho_s), np.ones_like(MC_rho))

#Make sure everything is fine
r_cut[r_cut < 0.0] = 0.0*r_cut[r_cut < 0.0] + MC_r[r_cut < 0.0]
r_cut[np.isnan(r_cut)] = MC_r[np.isnan(r_cut)]
r_cut[np.isinf(r_cut)] = MC_r[np.isinf(r_cut)]

#Impact parameter
b     = np.sqrt(np.random.uniform(0.0, r_cut**2, Ne))
L0    = 2.0*np.sqrt(r_cut**2-b**2) # Transversed length of the NS in the AMC in pc

Tenc  = L0/ut # Duration of the encounter in s
fa = 0.0755**2/ma
signal, BW  = NE.signal_isotropic(Bfld, Prd, MC_rho, fa, ut, s0, r_cut/MC_r, ret_bandwidth=True, profile=PROFILE)



for k in tqdm(range(Ne)):
    #Time series for the encounter
    tarray = np.linspace(0, Tenc[k], 1000)
    
    #Distance traversed during the encounter
    deltaL = L0[k]/2 - ut[k]*tarray
    
    # The MC profile is ~r^-9/4, so at each step the flux changes by (r/r+Delr)^-9/4
    rvals = np.sqrt(b[k]**2 + deltaL**2)
    if (PROFILE == "PL"):
        fact = (rvals/rvals[0])**(-9/4)
    elif (PROFILE == "NFW"):
        fact = 1/(rvals*(1+rvals)**2)
        fact /= fact[0]
    
    Signal_interaction = signal[k]*fact
    
    mean_flux = np.trapz(Signal_interaction, tarray)/Tenc[k]
    interaction_params = [s0[k], lG[k], bG[k], Tenc[k], mean_flux, MC_rho[k], MC_r[k]]
    Interactions.append(interaction_params)
    # SJW version
    interaction_params_SJW = [
        s0[k],
        lG[k],
        bG[k],
        MC_rho[k],
        MC_r[k],
        b[k],
        Prd[k],
        Bfld[k],
        theta[k],
        ut[k],
    ]
    Interactions_SJW.append(interaction_params_SJW)

Interactions = np.array(Interactions)
Interactions_SJW = np.array(Interactions_SJW)
#print(Interactions, Interactions.shape)


pert_text = ''
if (UNPERTURBED):
    pert_text = '_unperturbed'

int_file = dirs.data_dir + 'Interaction_params_%s%s%s%s%s.txt.gz'%(PROFILE, circ_text, cut_text, pert_text,IDstr)
int_file = dirs.data_dir + 'Interaction_params_SJW_%s%s%s%s%s.txt.gz'%(PROFILE, circ_text, cut_text, pert_text,IDstr)
    
print("Outputting to file:", int_file)

np.savetxt(
    int_file, 
    Interactions,
    header="Distance [pc], Galactic Longitude [deg], Galactic Latitude [deg], Length of encounter [s], Mean Flux [muJy], MC density [Msun/pc^3], MC radius [pc]", 
    fmt='%.5e'
)

np.savetxt(
    int_file,
    Interactions_SJW,
    header="Distance [pc], Galactic Longitude [deg], Galactic Latitude [deg], MC density [Msun/pc^3], MC radius [pc], Impact parameter [pc], Period [s], Bfield [Gauss], Viewing Angle [rad], Relative velocity [pc/s]",
    fmt="%.5e",
)
