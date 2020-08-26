#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.special as special
from scipy.special import erf
from scipy import interpolate
from scipy import linspace
import AMC
import NSencounter as NE
import perturbations as PB
import orbits
import glob
from tqdm import tqdm
import argparse
import sys
import os
import re

###
###  This code returns the simulated signal from axion MC - NS encounter
###

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_size = comm.Get_size()
    MPI_rank = comm.Get_rank()
    if (MPI_size > 1): USING_MPI = True
    
except (ModuleNotFoundError, ImportError) as err:
    print("   mpi4py module not found: using a single process only...")
    USING_MPI = False
    MPI_size = 1
    MPI_rank = 0

print(MPI_size, MPI_rank)


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


ma     = 2e-5*1e-9   # axion mass in GeV
maHz   = ma/hbar        # axion mass in GHz


bandwidth0 = vrel0**2/(2.*np.pi)*maHz*1.e9 # Bandwidth in Hz
maeV   = ma*1.e9        # axion mass in eV
min_enhancement = 1e-1    #Threshold for 'interesting' encounters (k = 10% of the local DM density)

f_AMC = 1.0 #Fraction of Dark Matter in the form of axion miniclusters

#Velocity dispersion: PB.sigma(R)*(3.24078e-14)

M_cut  = 1.0e-25
Phicut = 1.0e-3

#R_cut  = 7.19e-8  # GM_NS/u_dispersion^2 in pc

mmin = PB.M_min(maeV)
mmax = PB.M_max(maeV)
gg   = 1.7

## Neutron Star characteristics
MNS = 1.4     # MSun
RNS = 3.24e-13 # pc -- This corresponds to 10km
Pm = -0.3*np.log(10)  # Period peak
Ps = 0.34 # Spread of period
Bm = 12.65*np.log(10) # B field Peak peak
Bs = 0.55 # B field Peak spread

#TO BE DELETED:

# dispersion of NS in the disc
#sz = 1.0e3  # pc
#ss = 5.0e3  # pc
#st = 1./np.sqrt(0.5/ss**2 + 1/sz**2)

#alpEM  = 1/137.036      # Fine-structure constant
#ga     = alpEM/(2.*np.pi*fa)*(2./3.)*(4. + 0.48)/1.48 # axion-photon coupling in GeV^-1

parser = argparse.ArgumentParser(description='...')

parser.add_argument('-profile','--profile', help='Density profile for AMCs - `NFW` or `PL`', type=str, default="PL")
parser.add_argument('-unperturbed', '--unperturbed', help='Calculate for unperturbed profiles?', type=int, default=0)


args = parser.parse_args()
if (args.unperturbed <= 0):
    UNPERTURBED = False
else: 
    UNPERTURBED = True
    
PROFILE = args.profile



#abs_path  = '/Users/visinelli/Dropbox/LucaVisinelliArticoli/Axion star radio detection/current/data/'
plt_path = "../plots/"
abs_path = "../data_ecc/"

dist_path = abs_path+"distributions/"

#Load in survival probabilities
#a_surv_file = abs_path+"SurvivalProbability_a_" + PROFILE + ".txt" #List of survival probabilities
R_surv_file = abs_path+"SurvivalProbability_R_" + PROFILE + ".txt" #List of survival probabilities
#a_list, psurv_a_list = np.loadtxt(a_surv_file, delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
R_list, psurv_R_list = np.loadtxt(R_surv_file, delimiter =',', dtype='f8', usecols=(0,1), unpack=True)


#Load in encounter rates
if (UNPERTURBED):
    encounter_file =  abs_path+"EncounterRate_" + PROFILE + "_circ_unperturbed.txt" #List of encounter rates
else:
    encounter_file =  abs_path+"EncounterRate_" + PROFILE + ".txt" #List of encounter rates
R_list, dGammadR_list = np.loadtxt(encounter_file, delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
dGammadR_list *= f_AMC

#Generate some interpolation functions
#psurv_a     = interpolate.interp1d(a_list, psurv_a_list) # survival probability (as a function of a)
#psurv_R     = interpolate.interp1d(R_list, psurv_R_list) # survival probability (as a function of R)
dGammadR    = interpolate.interp1d(R_list, dGammadR_list)  # PDF of the galactic radius


dist_r_list = []
dist_rho_list = []
#Rdistr_list = []
# dGammadR_list  = []
# GalRadius_list = []

dict_interp_r = dict()
dict_interp_r_corr = dict()
dict_interp_rho = dict()
dict_interp_z   = dict()
dict_interp_mass = dict()



#Generate P(v|r)
# v = 
#def P_v_given_r(v, r):
    

# --------------------- First we prepare the sampling distributions and total interactions

if (UNPERTURBED):
    dist_r, dist_Pr, dist_Pr_sigu  = np.loadtxt(dist_path + 'distribution_radius_%s_unperturbed.txt'%(PROFILE,), delimiter =', ', dtype='f8', usecols=(0,1,2), unpack=True)
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
            distRX, distRY, distRC = np.loadtxt(dist_path + 'distribution_radius_%.2f_%s.txt'%(R_kpc, PROFILE), delimiter =', ', dtype='f8', usecols=(0,1,2), unpack=True)
            #distDX, distDY = np.loadtxt(dist_path + 'distribution_rho_%.2f_%s.txt'%(R_kpc, PROFILE), delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)
    
            #print("Here.")
            dist_r_list.append(distRX)
            #dist_rho_list.append(distDX)
            dict_interp_r[i] = interpolate.interp1d(distRX, distRY) # Radius distribution
            dict_interp_r_corr[i] = interpolate.interp1d(distRX, distRC) # Corrected radius distr 
            #print(np.trapz(distRC, distRX))
            #dict_interp_rho[i] = interpolate.interp1d(distDX[distDY>0.0], distDY[distDY>0.0], bounds_error=False, fill_value=(0, 0)) # Density distribution dPdrho
        
            if (PROFILE == "NFW"):
                distMX, distMY = np.loadtxt(dist_path + 'distribution_mass_%.2f_%s.txt'%(R_kpc, PROFILE), delimiter =', ', unpack=True)
                dict_interp_mass[i] = interpolate.interp1d(distMX, distMY, bounds_error=False, fill_value=0.0)
        
        except:
            print("Warning: File for radius %.2f does not exist" % R_kpc)
            dist_r_list.append(None)
            dist_rho_list.append(None)
            dict_interp_r[i] = None
            dict_interp_r_corr[i] = None
            #dict_interp_rho[i] = None
            
            if (PROFILE == "NFW"):
                dict_interp_mass[i] = None
            
            continue
        # Only interpolate for non-zeros

        #BJK: This is still the 'old' calculation
        # <sigma u> in pc^3/s
        # sigmau = np.trapz(distRX**2*distRY, distRX) * u_dispersion * np.sqrt(8*np.pi)
        # GC distance position
        # k = np.where(Rvals_distr == R)[0][0]
 
        # n_dist = NE.nNS_sph(R)     # NS distribution at R in pc^-3
        # n_NFW  = NE.rhoNFW(R)/Mavg # MC distribution at R in pc^-3
        # Del    = 4.0*np.pi*R**2    # differential volume in pc^2
        # integrand = n_dist*n_NFW*psurv(R)*Del*sigmau # integrand in s^-1 pc^-1

        # dGammadR_list.append(integrand)

# GalRadius_list, dGammadR_list = zip(*sorted(zip(Rdistr_list, dGammadR_list)))
# GammaTot = np.trapz(dGammadR_list, GalRadius_list)
GammaTot = np.trapz(dGammadR_list, R_list) # s^-1
print("Gamma = ", GammaTot, "s^-1 [for f_AMC = 1]")



# --------------------- Below we are computing the Signal


day    = 86400. # s
hour = 60*60 # s
DelT   = 1*day  # time step in seconds
RateT  = GammaTot*DelT # per 5 days
print(RateT)
skip   = 100
# Ntot   = int(skip + 109575*day/DelT) # skip + 300yrs total number of time steps

#BJK:OLD
Ntot   = int(skip + 200000*day/DelT)
#Ntot = 1e6

#Ntot   = int(skip + 200000*day/DelT) # FIXME: Testing on 1 year
TimTot = Ntot*DelT

PhiTot = []      # Time series of the total flux
L0tot  = []      # total length transversed by the NS
b2tot  = []      # internal variable used to compute the flux
# time_array = []  # time elapsed
sign_array = []  # signal at Earth in muJy, sum of PhiTot at each timestep
PhiMax     = []  # Maximum flux at each timestep 
LonMax     = []  # Longitude of maximum flux at each timestep
LatMax     = []  # Latitude of maximum flux at each timestep
lMax     = []
bMax     = []
Distnc   = []

ind  = -1
time = 0
time_array = np.linspace(0, Ntot*DelT, Ntot)
Signal_array = []
Interactions = []

#Ne = int(1e3)
Ne = int(1e6)


Prd   = np.random.lognormal(Pm, Ps, Ne) # Period of the NS in s^-1
Bfld  = np.random.lognormal(Bm, Bs, Ne) # B field of NS in gauss
theta = np.random.uniform(-np.pi/2.,np.pi/2., Ne)   # B field orientation FIXME: misalignment angle?

R_sample = NE.inverse_transform_sampling_log(dGammadR,  [np.min(R_list[psurv_R_list > 2e-6]), np.max(R_list)],  n_samples=Ne) # Galactocentric radius of interaction in pc
R_sample = np.array(R_sample)

# Draw the radii and densities of the axion MCs
Z_gal = np.zeros(Ne)
MC_r = np.zeros(Ne)
MC_rho = np.zeros(Ne)


    
#This is where you would parallelise it

for l, R in enumerate(tqdm(R_sample)):
    
    # Draw a value of z for the encounter
    # Zaxis = np.geomspace(R/1000, R, num=25, endpoint=True) # FIXME: Check binning
    # Zaxis = np.concatenate((-Zaxis, Zaxis), axis=None)
    dpdz  = lambda Z: NE.dPdZ(R, Z) 
    # dpdz  = dpdz/np.trapz(dpdz, Zaxis)
    # Zaxis, dpdz = zip(*sorted(zip(Zaxis, dpdz)))
    # Z_gal[l] = NE.inverse_transform_sampling(interpolate.interp1d(Zaxis, dpdz), Zaxis, n_samples=1)
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
        
        if (PROFILE == "NFW"):
            interp_M = dict_interp_mass[smallest]

        
    # radius in pc
    #CHECK THIS...
    MC_r[l] = NE.inverse_transform_sampling_log(interp_r_corr, dist_r, n_samples=1, nbins=1000)
    # TE Note: I added additional bins here and made it a log spaced in the inverse_transform_sampling
    # FIXME: Need to check if this change in inverse sampling is correct
    # More bins added to the inverse sampler too
    
    #PR = dict_interp_r_corr[smallest](MC_r[l]) # dPdr at a given galactocentric radius R
    # BJK: Note that this is *NOT* dPdr, this is dP_enc/dr
    
    # FIXME: distDY must be a mistake
    
    #BJK: this should be P(rho|r) = P(r|rho)P(rho)/P(r) = (3M/r) P(M) P(rho)/P(r)
    #RhoChR = NE.P_r_given_rho(MCrad[l], distDX, mmin, mmax, gg)*distDY/PR
    
    #BJK: My version
    # print(NE.P_r_given_rho(MCrad[l], distDX, mmin, mmax, gg), dict_interp_rho[smallst](distDX), dict_interp_rad[smallst](MCrad[l]))
    # quit()
    #P_rho_given_r = lambda rho: (4*np.pi*rho*MCrad[l]**2)*dict_interp_mass[smallst]((4*np.pi/3)*rho*MCrad[l]**3)/dict_interp_rad[smallst](MCrad[l])
    
    #P(rho|r) = (M/rho)*P(M)
    #Mf = (4*np.pi/3)*MC_r[l]**3
    
    #CHECK
    #rho_min = 1e-6
    rho_max = 3*mmax/(4*np.pi*MC_r[l]**3)
    if (UNPERTURBED or (PROFILE == "PL")):
        
        dPdM = lambda x: NE.HMF_sc(x, mmin, mmax, gg)/x
        #(M_f/rho)*P(M_f)
        P_rho_given_r = lambda rho: ((4*np.pi/3)*MC_r[l]**3)*dPdM((4*np.pi/3)*rho*MC_r[l]**3)
        rho_min = 3*mmin/(4*np.pi*MC_r[l]**3)
        
    else:
        P_rho_given_r = lambda rho: ((4*np.pi/3)*MC_r[l]**3)*interp_M((4*np.pi/3)*rho*MC_r[l]**3)
        rho_min = 3*(1e-3*mmin)/(4*np.pi*MC_r[l]**3)
    #P_rho_given_r = lambda rho: NE.P_r_given_rho(MCrad[l], rho, mmin, mmax, gg)*dict_interp_rho[smallst](rho)/dict_interp_rad[smallst](MCrad[l])

    
    #disDX  = distDX[RhoChR != 0]
    #RhoChR = RhoChR[RhoChR != 0]

    # MCrho is the AMC density in MSun/pc^3
    #if len(RhoChR)==0:
    #    MCrho[l] = 1.e-30
    #elif len(RhoChR)==1:
    #    MCrho[l] = disDX
    #else:
    #    RhoChR = RhoChR / np.trapz(RhoChR, disDX)
    # print(distDX, P_rho_given_r(distDX))
    # quit()
    
    
    MC_rho[l] = NE.inverse_transform_sampling_log(P_rho_given_r, [rho_min, rho_max], n_samples=1, nbins=1000)
    #print(MC_rho[l])
    # TE Note: I added additional bins here and made it a log spaced in the inverse_transform_sampling
    # FIXME: Need to check if this change in inverse sampling is correct
    # I also added a small value to HMF_sc(mass, mmin, mmax, gg): since it was numerically unstable at 0.0
    # More bins added to the inverse sampler too

    # if np.isnan(MCrho[l]):
    #     print('heres the problem', NE.P_r_given_rho(MCrad[l], distDX, mmin, mmax, gg))
    #     plt.loglog(distDX, NE.P_r_given_rho(MCrad[l], distDX, mmin, mmax, gg))
    #     plt.xlabel(r'Mass')
    #     plt.ylabel(r'$P(r|r\rho$)')
    #     plt.savefig('../plots/dPdr_given_rho.pdf')
    #     quit()

psi   = np.random.uniform(-np.pi,np.pi,Ne)  # Longitude
xi    = np.arcsin(Z_gal/R_sample)              # Latitude
# MCdis = 1.0*MCrad # Initial distance of the NS from the centre of the AMC in pc
#MCvel = NE.Vcirc(MCorb)  # pc/s

#VNSX  = MCvel*np.cos(psi)
#VNSY  = MCvel*np.sin(psi)
z0    = Z_gal
x0    = R_sample*np.cos(xi)*np.cos(psi)
y0    = R_sample*np.cos(xi)*np.sin(psi)

s0    = np.sqrt((RSun + x0)**2 + y0**2 + z0**2) # Distance from events in pc
bG    = np.arctan(z0/np.sqrt((RSun + x0)**2 + y0**2))*180.0/np.pi # Galactic Latitude
lG    = np.arctan(y0/(RSun + x0))*180.0/np.pi # Galactic Longitude

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
# print(rho_crit,MCrad, MCrho)


if (PROFILE == "PL"):
    r_cut = MC_r*np.minimum((MC_rho/(4*rho_crit))**(4/9), np.ones_like(MC_rho))
    
elif (PROFILE == "NFW"):
    c = 100
    rho_s = MC_rho*(c**3/(3*NE.f_NFW(c))) #Convert mean density rhoi to AMC density
    #print(rho_crit/rho_AMC)
    #print(c**3/(3*NE.f_NFW(c)))
    #print(rho_crit/rho_AMC)
    r_cut = MC_r*np.minimum(NE.x_of_rho(rho_crit/rho_s), np.ones_like(MC_rho))
    #r_cut += 1e-15
    #print(r_cut)
    

b     = np.sqrt(np.random.uniform(0.0, r_cut**2, Ne))
L0    = 2.0*np.sqrt(r_cut**2-b**2) # Transversed length of the NS in the AMC in pc
b2tot = np.append(b2tot, b**2)
L0tot = np.append(L0tot, L0)


Tenc  = L0/ut # Duration of the encounter in s
fa = 0.0755**2/ma
signal, BW  = NE.signal_isotropic(Bfld, Prd, MC_rho, fa, ut, s0, r_cut/MC_r, ret_bandwidth=True, profile=PROFILE) # FIXME: Check that this is correct

#print("R, r_cut, b:", MCrad, r_cut, b)
#print("Tenc, L0, ut, M, MCrho, flux_i:", Tenc, L0, ut, 4/3*np.pi*MCrho*MCrad**3., MCrho, signal)
#print(" ")
# PhiTot = np.append(PhiTot, sign)
# J = np.argmax(sign) # brightest new encounter in that timestep

for k in tqdm(range(Ne)):
    # print(i + int(Tenc[k]/DelT), Ntot-1)
    #i = int(np.random.rand()*Ntot)
    #end_index = min(i + int(Tenc[k]/DelT), Ntot-1)+1
    # TE Note: Added +1 at the end here for events that were less than one time step
    #deltaL = L0[k]/2 - ut[k]*time_array[i:end_index]
    
    #deltaL = L0[k]/2 - ut[k]*(time_array[i:end_index] - time_array[i])
    
    #BJK: NEW
    tarray = np.linspace(0, Tenc[k], 1000)
    deltaL = L0[k]/2 - ut[k]*tarray
    
    #BJK: OLD
    #i = int(np.floor(np.random.rand()*Ntot))
    #end_index = min(i + int(Tenc[k]/DelT), Ntot-1)+1
    # TE Note: Added +1 at the end here for events that were less than one time step
    #deltaL = L0[k]/2 - ut[k]*time_array[i:end_index]
    #deltaL = L0[k]/2 - ut[k]*(time_array[i:end_index] - time_array[i])
    

    
    # print(i,end_index)
    
    # The MC profile is ~r^-9/4, so at each step the flux changes by (r/r+Delr)^-9/4
    rvals = np.sqrt(b[k]**2 + deltaL**2)
    if (PROFILE == "PL"):
        fact = (rvals/rvals[0])**(-9/4)
    elif (PROFILE == "NFW"):
        fact = 1/(rvals*(1+rvals)**2)
        fact /= fact[0]
    
    #L0tot1 = L0tot - vrel*DelT # New positions of the MCs

    #fact = ((L0tot1**2 + b2tot)/(L0tot**2 + b2tot))**(-9./8.)
    #PhiTot = np.multiply(PhiTot, fact) # update the flux vector
    
    Signal_interaction = signal[k]*fact
    
    #Signal_interaction = np.concatenate((np.zeros(i-1), Signal_interaction))
    #Signal_interaction = np.concatenate((Signal_interaction, np.zeros(Ntot - Signal_interaction.size)))
    #Signal_array.append(Signal_interaction)

    #BJK: Added peak flux here
    mean_flux = np.trapz(Signal_interaction, tarray)/Tenc[k]
    interaction_params = [s0[k], lG[k], bG[k], BW[k], Tenc[k], np.max(Signal_interaction), mean_flux, MC_rho[k], MC_r[k], R_sample[k]]
    Interactions.append(interaction_params)

Interactions = np.array(Interactions)
#print(Interactions, Interactions.shape)
Signal_array = np.array(Signal_array)
time_array = np.array(time_array)

pert_text = ''
if (UNPERTURBED):
    pert_text = '_unperturbed'


int_file = abs_path + 'Interaction_params_%s%s.txt'%(PROFILE, pert_text)

#tag = 1
#while os.path.isfile(int_file):
#    int_file = abs_path + 'Interaction_params_%s%s_%d.txt'%(PROFILE, pert_text, tag)
#    tag += 1
    
print("Outputting to file:", int_file)

np.savetxt(int_file, Interactions,
        header="Distance [pc], Galactic Longitude [deg], Galactic Latitude [deg], Bandwidth [Hz], Legnth of encounter [s], Peak Flux [muJy], Mean Flux [muJy], MC density [Msun/pc^3], MC radius [pc], galactocentric radius [pc]")

print("NB: Currently not printing signal arrays due to huge file length for long simulations (see also line 377...)")
#np.savetxt(abs_path + 'time_array.txt', time_array, header="Times [s]")
#np.savetxt(abs_path + 'Signal_array.txt', Signal_array, header="Signals [muJy]")



        # L0tot1 = L0tot - vrel*DelT # New positions of the MCs
        # # The MC profile is ~r^-9/4, so at each step the flux changes by (r/r+Delr)^-9/4
        # fact = ((L0tot1**2 + b2tot)/(L0tot**2 + b2tot))**(-9./8.)
        # PhiTot = np.multiply(PhiTot, fact) # update the flux vector
        # L0tot = L0tot1 # update the position vector


        # FIXME: Does this only accept the brightest new encounter per timestep?
        # FIXME: Currently we ignore intrinsic velocity despersion variation as a funciton of radius
        
        # LatMax = np.append(LatMax, xi[J])
        # LonMax = np.append(LonMax, psi[J])
        # bMax   = np.append(bMax,   bG[J])
        # lMax   = np.append(lMax,   lG[J])
        # PhiMax = np.append(PhiMax, sign[J]) 
        # Distnc = np.append(Distnc, s0[J])

# 

"""
for event in events:
    times = np.linspace(event.t_start, event.t_end, 100)
    fluxes = event.phi(times)

    massive_t_list.append(times)
    massive_phi_list.append(fluxes)

"""

# LatMax = LatMax[PhiMax>Phicut]
# LonMax = LonMax[PhiMax>Phicut]
# bMax   = bMax[PhiMax>Phicut]
# lMax   = lMax[PhiMax>Phicut]
# Distnc = Distnc[PhiMax>Phicut]
# PhiMax = PhiMax[PhiMax>Phicut]

# time_array[sign_array==0] = 0
# time_array = np.trim_zeros(time_array)
# sign_array = np.trim_zeros(sign_array)
# time_array = time_array[skip:]
# sign_array = sign_array[skip:]

# np.savetxt(abs_path + 'Skyplots.txt', np.column_stack([LatMax, LonMax, bMax, lMax, Distnc, PhiMax]), delimiter=', ', header="LatMax, LonMax, bMax, lMax, Distance [pc], PhiMax [muJy]")

# np.savetxt(abs_path + 'Timeplots.txt', np.column_stack([time_array, sign_array]), delimiter=', ', header="time_array, sign_array")

### Below is just plotting

# exit()

# Nbins = 20

# hist, bins, _ = plt.hist(PhiMax, bins=Nbins)
# logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
# plt.close('all')
# plt.hist(PhiMax, bins=logbins)
# plt.xscale('log')
# #plt.show(block=False)
# plt.savefig('/Users/visinelli/Desktop/Results/histsignal.pdf')

# plt.close('all')
# plt.hist2d(LonMax, LatMax, bins=(Nbins, Nbins), cmap=plt.cm.jet)
# #plt.show(block=False)
# plt.savefig('/Users/visinelli/Desktop/Results/GalSkyMap.pdf')

# plt.close('all')
# plt.hist(LatMax, bins=Nbins)
# plt.hist(LonMax, bins=Nbins)
# plt.savefig('/Users/visinelli/Desktop/Results/Galdistribution.pdf')


# plt.close('all')
# plt.hist(bMax, bins=Nbins)
# plt.hist(lMax, bins=Nbins)
# #plt.show(block=False)
# plt.savefig('/Users/visinelli/Desktop/Results/skymap.pdf')

# plt.close('all')
# plt.hist2d(bMax, lMax, bins=(Nbins, Nbins), cmap=plt.cm.jet)
# #plt.show(block=False)
# plt.savefig('/Users/visinelli/Desktop/Results/skydistribution.pdf')

# #plt.close('all')
# #Phi_edges  = np.geomspace(np.min(PhiMax), np.max(PhiMax), 50)
# #Phi_centre = np.sqrt(Phi_edges[1:] * Phi_edges[:-1])
# #Phi_bin    = np.histogram(PhiMax, Phi_edges, density=True)[0]
# #plt.hist(Phi_bin, bins=np.log10(Phi_centre))
# #plt.show(block=False)
# #plt.savefig('/Users/visinelli/Desktop/Results/signal.pdf')
# #exit()

# time_array[sign_array==0] = 0
# time_array = np.trim_zeros(time_array)
# sign_array = np.trim_zeros(sign_array)
# time_array = time_array[skip:]
# sign_array = sign_array[skip:]

# plt.close('all')
# plt.plot(time_array, np.log10(sign_array), 'b-')
# #plt.show(block=False)
# plt.savefig('/Users/visinelli/Desktop/Results/signal.pdf')
