#!/usr/bin/env python3
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

from tqdm import tqdm
import argparse
import sys 
import os

import NSencounter as NE
import perturbations as PB
import mass_function
import params
import dirs
import tools
from tools import r_AS

import MilkyWay
import Andromeda

if not os.path.exists(dirs.data_dir + "distributions/"):
    os.makedirs(dirs.data_dir + "distributions/")


# constants
hbar = 6.582e-16  # GeV/GHz
Tage = 4.26e17
RSun = 8.33e3  # pc
pc = 3.086e16  # pc in m
cs = 3.0e8  # speed of light in m/s
vrel0 = 1.0e-3
vrel = vrel0 * cs / pc  # relative velocity in pc/s
u_dispersion = 1.0e-11  # velocity dispersion in pc/s



M_cut = 1.0e-29

## Neutron Star characteristics
MNS = 1.4  # MSun
RNS = 3.24e-13  # pc -- This corresponds to 10km

Nbins_mass = 300



# ---------------------------------
# -------- OPTIONS ----------------
# ---------------------------------

def sample_encounters(Ne, m_a, profile,  AMC_MF, galaxyID = "MW", circular=False, AScut = False, unperturbed=False, IDstr=""):

    Ne = int(Ne)

    file_suffix_in = tools.generate_suffix(profile, AMC_MF, circular=circular,
                                        AScut=False, unperturbed=unperturbed, IDstr=IDstr, verbose=False)
                                        
    #We need different labels for the input and output files, because
    #we may also need to label the outputs with "AScut" if we use that option
    file_suffix_out = tools.generate_suffix(profile, AMC_MF, circular=circular,
                                        AScut=AScut, unperturbed=unperturbed, IDstr=IDstr, verbose=True)
    
    #Set up the mass function
    M0 = AMC_MF.M0

    if (galaxyID == "MW"):
        Galaxy = MilkyWay
    elif (galaxyID == "M31"):
        Galaxy = Andromeda
    else:
        raise ValueError("Invalid galaxyID.")


    plt_path = dirs.plot_dir
    dist_path = dirs.data_dir + "distributions/"

    #Select which columns we want, depending on whether we're including the AS cut
    if (AScut):
        col = 2
    else:
        col = 1

    # Load in survival probabilities
    R_surv_file = dirs.data_dir + "SurvivalProbability_R_" + file_suffix_in + ".txt"# List of survival probabilities    
    R_list, psurv_R_list = np.loadtxt(R_surv_file, delimiter=",", dtype="f8", usecols=(0, col), unpack=True)

    # Load in encounter rates
    encounter_file = dirs.data_dir + "EncounterRate_" + file_suffix_in + ".txt"
    R_list, dGammadR_list = np.loadtxt(encounter_file, delimiter=",", dtype="f8", usecols=(0, col), unpack=True)

    # Generate some interpolation functions
    # psurv_a     = interpolate.interp1d(a_list, psurv_a_list) # survival probability (as a function of a)
    # psurv_R     = interpolate.interp1d(R_list, psurv_R_list) # survival probability (as a function of R)
    dGammadR = interpolate.interp1d(R_list, dGammadR_list)  # PDF of the galactic radius

    #Set up dictionaries, which will hold the M, R, rho distributions of AMCs at each radius
    dist_r_list = []
    dist_rho_list = []

    dict_interp_r = dict()
    dict_interp_r_corr = dict()
    dict_interp_rho = dict()
    dict_interp_z = dict()
    dict_interp_mass = dict()

    # --------------------- First we prepare the sampling distributions and total interactions

    if unperturbed:
        dist_r, dist_Pr, dist_Pr_sigu = np.loadtxt( dist_path + "distribution_radius_" + file_suffix_in + ".txt", delimiter=", ", dtype="f8", usecols=(0, 1, 2), unpack=True)

        interp_r = interpolate.interp1d(dist_r, dist_Pr)
        interp_r_corr = interpolate.interp1d(dist_r, dist_Pr_sigu)

    else:

        # Loop through distances
        for i, R in enumerate(R_list):
            #R_kpc = R / 1e3

            # distRX is AMC radius
            # distRY is the PDF dP/dR (normalised as int dP/dR dR = 1) where R is the AMC radius
            # distRC is the PDF <sigma u>*dP/dR where R is the AMC radius
            # distDX is the list of densities for dPdrho
            # distDY is the dPdrho

            try:
                #print("Loading file for radius %.4f " % (R,))
                fname = dist_path + f"distribution_radius_{R:.4f}_{file_suffix_out}.txt"
                #print(fname)
                distRX, distRY, distRC = np.loadtxt(
                    fname,
                    delimiter=", ",
                    dtype="f8",
                    usecols=(0, 1, 2),
                    unpack=True,
                )
                # distDX, distDY = np.loadtxt(dist_path + 'distribution_rho_%.2f_%s.txt'%(R_kpc, PROFILE), delimiter =', ', dtype='f8', usecols=(0,1), unpack=True)

                dist_r_list.append(distRX)
                dict_interp_r[i] = interpolate.interp1d(
                    distRX, distRY
                )  # Radius distribution
                dict_interp_r_corr[i] = interpolate.interp1d(
                    distRX, distRC
                )  # Corrected radius distr
        
                fname = dist_path + f"distribution_mass_{R:.4f}_{file_suffix_out}.txt"
                #print(fname)
                distMX, distMY = np.loadtxt(
                    fname,
                    delimiter=", ",
                    unpack=True,
                )
                dict_interp_mass[i] = interpolate.interp1d(
                    distMX, distMY, bounds_error=False, fill_value=0.0
                )

            except:
                #print("Warning: File for radius %.4f does not exist" % (R,))
                dist_r_list.append(None)
                dist_rho_list.append(None)
                dict_interp_r[i] = None
                dict_interp_r_corr[i] = None
                # dict_interp_rho[i] = None

                # if (PROFILE == "NFW"):
                dict_interp_mass[i] = None

                continue

    GammaTot = np.trapz(dGammadR_list, R_list)  # s^-1
    #print("Gamma = ", GammaTot, "s^-1 [for f_AMC = 1]")


    # --------------------- Below we are computing the Signal

    # BJK: This still needs lots of cleaning up! A lot of this may be left-over useless stuff.

    #BJK: GOT TO HERE!!!
    Interactions = []

    R_sample = tools.inverse_transform_sampling(
        dGammadR, [np.min(R_list[psurv_R_list > 2e-6]), np.max(R_list)], n_samples=Ne, logarithmic=True
    )  # Galactocentric radius of interaction in pc
    R_sample = np.array(R_sample)

    # Draw the radii and densities of the axion MCs
    Z_gal = np.zeros(Ne)
    MC_r = np.zeros(Ne)
    MC_rho = np.zeros(Ne)


    for l, R in enumerate(tqdm(R_sample, desc="> Sampling encounters")):

        # Draw a value of z for the encounter
        dpdz = lambda Z: Galaxy.dPdZ(Z)
        Z_gal[l] = tools.inverse_transform_sampling(dpdz, np.array([-R, R]), n_samples=1, nbins=1000)

        Pr_check = 0
        if not unperturbed:
            # Find the GC radius that is closer to what is drawn from distribution
            abs_val = np.abs(R_list - R)
            smallest = abs_val.argmin()
            while (Pr_check <= 0):
                R_orb = R_list[smallest]  # Closest orbit to the r drawn
                dist_r = dist_r_list[smallest]  # List of MC radii
                # dist_rho  = dist_rho_list[smallest] # List of MC densities
                # interp_rho = dict_interp_rho[smallest]
                interp_r_corr = dict_interp_r_corr[smallest]

                # if (PROFILE == "NFW"):
                interp_M = dict_interp_mass[smallest]
                Pr_check = np.sum(interp_r_corr(dist_r))
                if (Pr_check <= 0):
                    #print("Trying next radius...")
                    smallest -= 1
            
        
        alpha_AS = r_AS(1.0, m_a)
        k_AMC = (3 / (4 * np.pi)) ** (1 / 3)
    
        # radius in pc
        MC_r[l] = tools.inverse_transform_sampling(
            interp_r_corr, dist_r, n_samples=1, nbins=10000, logarithmic=True
        )

        rho_max = 3 * AMC_MF.mmax / (4 * np.pi * MC_r[l] ** 3)
    
        if AScut:
            rho_min_AS = (alpha_AS * k_AMC / MC_r[l] ** 2) ** 3
        else:
            rho_min_AS = 1e-30

        # Sample AMC density rho given the AMC radius R
        if unperturbed:

            dPdM = lambda x: AMC_MF.dPdlogM(x) / x
            # (M_f/rho)*P(M_f)
            P_rho_given_r = lambda rho: ((4 * np.pi / 3) * MC_r[l] ** 3) * dPdM(
                (4 * np.pi / 3) * rho * MC_r[l] ** 3
            )
            rho_min = np.maximum(3 * AMC_MF.mmin / (4 * np.pi * MC_r[l] ** 3), rho_min_AS)

        else:
            P_rho_given_r = lambda rho: ((4 * np.pi / 3) * MC_r[l] ** 3) * interp_M(
                (4 * np.pi / 3) * rho * MC_r[l] ** 3
            )
            #rho_min = np.maximum(
            #    3 * (1e-6 * AMC_MF.mmin) / (4 * np.pi * MC_r[l] ** 3), rho_min_AS
            #)
            rho_min = np.maximum(
                3 * (1e-10 * M0) / (4 * np.pi * MC_r[l] ** 3), rho_min_AS
            )
        # P_rho_given_r = lambda rho: NE.P_r_given_rho(MCrad[l], rho, mmin, mmax, gg)*dict_interp_rho[smallst](rho)/dict_interp_rad[smallst](MCrad[l])

        MC_rho[l] = tools.inverse_transform_sampling(
            P_rho_given_r, [1e-6*rho_min, rho_max], n_samples=1, nbins=100000, logarithmic=True
        )
    
        if (np.isnan(MC_rho[l])):
            print(R, R_orb, MC_r[l], MC_rho[l])
            print(dist_r)
            print(interp_r_corr(dist_r))
        
        #if (1 < 2):
            plt.figure()
            rho_list = np.geomspace(rho_min, rho_max, 100000)
            plt.loglog(rho_list, P_rho_given_r(rho_list))
            plt.show()

        # print(rho_min, MC_rho[l])


    psi = np.random.uniform(-np.pi, np.pi, Ne)  # Longitude
    xi = np.arcsin(Z_gal / R_sample)  # Latitude



    z0 = Z_gal
    x0 = R_sample * np.cos(xi) * np.cos(psi)
    y0 = R_sample * np.cos(xi) * np.sin(psi)
    R_spherical = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)

    s0 = np.sqrt((RSun + x0) ** 2 + y0 ** 2 + z0 ** 2)  # Distance from events in pc
    bG = (
        np.arctan(z0 / np.sqrt((RSun + x0) ** 2 + y0 ** 2)) * 180.0 / np.pi
    )  # Galactic Latitude
    lG = np.arctan2(y0, (RSun + x0)) * 180.0 / np.pi  # Galactic Longitude

    # Relative velocity between NS & AMC
    vrel = np.sqrt(2) * Galaxy.sigma(R_sample) * 3.24078e-14
    ux = np.random.normal(0, vrel, Ne)  # pc/s
    uy = np.random.normal(0, vrel, Ne)  # pc/s
    uz = np.random.normal(0, vrel, Ne)  # pc/s
    # ut    = np.abs(np.sqrt((ux + VNSX)**2 + (uy + VNSY)**2 + uz**2)) # pc/s
    ut = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)  # pc/s

    # BJK: Here, we need to make sure that we implement the same cut on the
    # impact parameter as we did for the radii of very diffuse AMCs
    rho_loc = Galaxy.rhoNFW(R_sample)
    rho_crit = rho_loc * params.min_enhancement


    if profile == "PL":
        r_cut = MC_r * np.minimum(
            (MC_rho / (4 * rho_crit)) ** (4 / 9), np.ones_like(MC_rho)
        )

    elif profile == "NFW":
        c = 100
        rho_s = MC_rho * (
            c ** 3 / (3 * NE.f_NFW(c))
        )  # Convert mean density rhoi to AMC density
        r_cut = MC_r * np.minimum(NE.x_of_rho(rho_crit / rho_s), np.ones_like(MC_rho))

    # Make sure everything is fine
    r_cut[r_cut < 0.0] = 0.0 * r_cut[r_cut < 0.0] + MC_r[r_cut < 0.0]
    r_cut[np.isnan(r_cut)] = MC_r[np.isnan(r_cut)]
    r_cut[np.isinf(r_cut)] = MC_r[np.isinf(r_cut)]

    # Impact parameter
    b = np.sqrt(np.random.uniform(0.0, r_cut ** 2, Ne))


    MC_M = (4*np.pi*MC_r**3/3)*MC_rho

    lablist = ["x0", "y0", "z0", "rho", "r", "M"]

    for k in range(Ne):
        #Select from either the GC population or the CMZ population
        if (R_spherical[k] > 10):
            NS_dict = Galaxy.CMZ_dict
            len_NS = Galaxy.len_CMZ
        else:
            NS_dict = Galaxy.GC_dict
            len_NS = Galaxy.len_GC
        
        iNS = np.random.randint(0, len_NS)
    
        # SJW version
        interaction_params = [
            NS_dict["B0"][iNS],
            NS_dict["T"][iNS],
            NS_dict["theta"][iNS],
            NS_dict["t"][iNS],
            x0[k],
            y0[k],
            z0[k],
            MC_rho[k],
            MC_r[k],
            MC_M[k],
            b[k],
            ut[k]
        ]
        Interactions.append(interaction_params)

    #Interactions = np.array(Interactions)
    Interactions = np.array(Interactions)
    # print(Interactions, Interactions.shape)

    int_file = dirs.data_dir + "Interaction_params_" + file_suffix_out + ".txt.gz"

    print("> Outputting encounters to file:", int_file)
   
    np.savetxt(
        int_file,
        Interactions,
        header="B0 [G], Period [s], Misalignment angle [radians], NS Age [Myr], x [pc], y [pc], z [pc], AMC density [Msun/pc^3], AMC radius [pc], AMC mass [Msun],  Impact parameter [pc], Relative velocity [pc/s]",
        fmt="%.5e",
    )
    return b/ut


def getOptions(args=sys.argv[1:]):
    # Parse the arguments!
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument("-Ne", "--Ne", help="Number of encounters to sample", type=int, default=1000)
    parser.add_argument("-profile", "--profile", help="Density profile for AMCs - `NFW` or `PL`", type=str, default="PL")
    parser.add_argument("-m_a", "--m_a", type=float, help="Axion mass in eV", default = 50e-6)
    parser.add_argument("-unperturbed", "--unperturbed", help="Calculate for unperturbed profiles?", type=bool, default=False)
    parser.add_argument("-galaxyID", "--galaxyID", type=str, help="ID of galaxy - 'MW' or 'M31'", default="MW")
    parser.add_argument("-circ", "--circular", dest="circular", action="store_true", help="Use the circular flag to force e = 0 for all orbits.")
    parser.add_argument("-AScut", "--AScut", dest="AScut", action="store_true", help="Include an axion star cut on the AMC properties.")
    parser.add_argument("-MF_ID", "--mass_function_ID", help="...", type=str, default="delta_c")
    parser.add_argument("-IDstr", "--IDstr", type=str, help = "ID string to label the output files.", default="")
    parser.set_defaults(circular=False)
    parser.set_defaults(AScut=False)

    options = parser.parse_args(args)
    return options
    
if __name__ == '__main__':
    opts = getOptions(sys.argv[1:])
    
    #Create a mass function based on the input "mass function ID"
    AMC_MF = get_mass_function(opts.MF_ID, opts.m_a, opts.profile)
    AMC_MF.label = opts.MF_ID

    sample_encounters(opts.Ne, opts.m_a, opts.profile, AMC_MF,  opts.galaxyID,  opts.circular, opts.AScut, opts.unperturbed, opts.IDstr)


