#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
# from scipy.integrate import quad
# from scipy import interpolate
# import AMC
import mass_function
import NSencounter as NE
import perturbations as PB
import glob

try:
    from tqdm import tqdm
except ImportError as err:

    def tqdm(x):
        return x


#galaxy_ID = "MW"
galaxy_ID = "M31"

if (galaxy_ID == "MW"):
    import MilkyWay as Galaxy
else:
    import Andromeda as Galaxy

import argparse
import sys
import os
import re
import warnings

import params

# sys.path.append("../")
import dirs

if not os.path.exists(dirs.data_dir + "distributions/"):
    os.makedirs(dirs.data_dir + "distributions/")

# The code in principle is parallelised, but I wouldn't recommend it...
USING_MPI = True

warnings.filterwarnings("error")


# This mass corresponds roughly to an axion decay
# constant of 3e11 and a confinement scale of Lambda = 0.076
in_maeV = params.m_a  # axion mass in eV
in_gg = -0.7

print("> Using m_a = %.2e eV, gamma = %.4f" % (in_maeV, in_gg))


######################
####   OPTIONS  ######

# Parse the arguments!
parser = argparse.ArgumentParser(description="...")

parser.add_argument(
    "-profile",
    "--profile",
    help="Density profile for AMCs - `NFW` or `PL`",
    type=str,
    default="PL",
)
parser.add_argument(
    "-unperturbed",
    "--unperturbed",
    help="Calculate for unperturbed profiles?",
    type=bool,
    default=False,
)
parser.add_argument(
    "-max_rows",
    "--max_rows",
    help="Maximum number of rows to read from each file?",
    type=int,
    default=None,
)
parser.add_argument(
    "-circ",
    "--circular",
    dest="circular",
    action="store_true",
    help="Use the circular flag to force e = 0 for all orbits.",
)
parser.add_argument(
    "-AScut",
    "--AScut",
    dest="AScut",
    action="store_true",
    help="Include an axion star cut on the AMC properties.",
)
parser.add_argument(
    "-mass_choice",
    "--mass_choice",
    help="Mass parameter = 'c' or 'a' for characteristic or average.",
    type=str,
    default="c",
)
parser.set_defaults(circular=False)
parser.set_defaults(AScut=False)

args = parser.parse_args()
UNPERTURBED = args.unperturbed
PROFILE = args.profile
CIRCULAR = args.circular
AS_CUT = args.AScut
max_rows = args.max_rows
MASS_CHOICE = args.mass_choice

circ_text = ""
if CIRCULAR:
    circ_text = "_circ"

cut_text = ""
if AS_CUT:
    print("> Calculating with axion-star cut...")
    cut_text = "_AScut"


if MASS_CHOICE.lower() == "c":
    M0 = mass_function.calc_Mchar(in_maeV)
elif ((MASS_CHOICE.lower() == "a") or (MASS_CHOICE.lower() == "full")):
    AMC_MF = mass_function.PowerLawMassFunction(m_a=in_maeV, gamma=in_gg, profile=PROFILE)
    M0 = AMC_MF.mavg
    
elif MASS_CHOICE.lower() == "p":
    M0 = 1e-14*(params.m_a/50e-6)**(-0.5)
    
print(M0)

#if PROFILE == "NFW" and UNPERTURBED == False:
    #M0 = mass_function.mass_after_stripping(M0)

# Mass function
if PROFILE == "PL" or UNPERTURBED == True:
    AMC_MF = mass_function.PowerLawMassFunction(m_a=in_maeV, gamma=in_gg, profile=PROFILE)
elif PROFILE == "NFW":
    print("IGNORING STRIPPING!")
    AMC_MF = mass_function.PowerLawMassFunction(m_a=in_maeV, gamma=in_gg, profile=PROFILE)
    #AMC_MF = mass_function.StrippedPowerLawMassFunction(m_a=in_maeV, gamma=in_gg)


M_cut = 1e-29

# IDstr = "_ma_57mueV"
IDstr = params.IDstr
IDstr_out = IDstr +  "_delta_" + MASS_CHOICE.lower()


Nbins_mass = 300
Nbins_radius = 1000  # Previously 500

# How much smaller than the local DM density
# do we care about?
k = params.min_enhancement

# Define AS cut
def r_AS(M_AMC):
    m_22 = in_maeV / 1e-22
    return 1e3 * (1.6 / m_22) * (M_AMC / 1e9) ** (-1 / 3)

print("Axion stars cut-off radius:", r_AS(M0))


alpha_AS = r_AS(1.0)
k_AMC = (3 / (4 * np.pi)) ** (1 / 3)


def prepare_distributions():
    
    

def main():

    print("ID string:", IDstr)
    a_grid = None
    if MPI_rank == 0:
        # Gather the list of files to be used, then loop over semi-major axis a
        f_search = dirs.montecarlo_dir + "AMC_samples_*" + PROFILE + circ_text +  IDstr + ".txt"
        ff1 = glob.glob(f_search)
        print(f_search)
 
        a_grid = np.zeros(len(ff1))

        for i, fname in enumerate(ff1):
            # print(fname)
            m = re.search("AMC_samples_a=(.+?)_" + PROFILE + circ_text + IDstr + ".txt", fname)
            if m:
                a_string = m.group(1)
            a_grid[i] = float(a_string) # conversion to pc

        a_grid = np.sort(a_grid)

    # Edges to use for the output bins in R (galactocentric radius, pc)
    if CIRCULAR:
        R_centres = 1.0 * a_grid
    else:
        R_bin_edges = np.geomspace(1e-2, 60e3, 101)
        #R_bin_edges = np.geomspace(1e-2, 10, 65)
        R_centres = np.sqrt(R_bin_edges[:-1] * R_bin_edges[1:])

    mass_ini_all, mass_all, radius_all, e_all, a_all = load_AMC_results(a_grid)

    # ----------------------------

    # Re-weight the samples according to radius
    if CIRCULAR:
        (
            AMC_weights,
            AMC_weights_surv,
            AMC_weights_masscut,
            AMC_weights_AScut,
            AMC_weights_AScut_masscut,
        ) = calculate_weights_circ(
            a_grid, a_all, e_all, mass_all, mass_ini_all, radius_all
        )
    else:
        (
            AMC_weights,
            AMC_weights_surv,
            AMC_weights_masscut,
            AMC_weights_AScut,
            AMC_weights_AScut_masscut,
        ) = calculate_weights(
            R_bin_edges, a_grid, a_all, e_all, mass_all, mass_ini_all, radius_all
        )  # Just pass the eccentricities and semi major axes



    # quit()

    if MPI_rank == 0:

        # Calculate the survival probability as a function of a
        psurv_a_list, psurv_a_AScut_list = calculate_survivalprobability(
            a_grid, a_all, mass_all, mass_ini_all, radius_all
        )

        P_r_weights = np.sum(
            AMC_weights, axis=0
        )  # Check if this should be a sum or integral
        P_r_weights_surv = np.sum(AMC_weights_surv, axis=0)
        P_r_weights_masscut = np.sum(AMC_weights_masscut, axis=0)
        P_r_weights_AScut = np.sum(AMC_weights_AScut, axis=0)
        P_r_weights_AScut_masscut = np.sum(AMC_weights_AScut_masscut, axis=0)

        psurv_R_list = P_r_weights_surv / (P_r_weights + 1e-30)

        # Save the outputs
        if not UNPERTURBED:
            # np.savetxt(output_dir + 'Rvals_distributions_' + PROFILE + '.txt', Rvals_distr)
            if not CIRCULAR:
                np.savetxt(
                    dirs.data_dir + "SurvivalProbability_a_" + PROFILE + IDstr_out + ".txt",
                    np.column_stack([a_grid, psurv_a_list, psurv_a_AScut_list]),
                    delimiter=", ",
                    header="Columns: semi-major axis [pc], survival probability, survival probability for AMCs passing the AS cut",
                )
            np.savetxt(
                dirs.data_dir
                + "SurvivalProbability_R_"
                + PROFILE
                + circ_text
                + IDstr_out
                + ".txt",
                np.column_stack(
                    [
                        R_centres,
                        psurv_R_list,
                        P_r_weights,
                        P_r_weights_surv,
                        P_r_weights_masscut,
                        P_r_weights_AScut,
                        P_r_weights_AScut_masscut,
                    ]
                ),
                delimiter=", ",
                header="Columns: galactocentric radius [pc], survival probability, Initial AMC density [Msun/pc^3], Surviving AMC density [Msun/pc^3], Surviving AMC density with mass-loss < 90% [Msun/pc^3], Surviving AMC density with R_AMC > R_AS [Msun/pc^3], Surviving AMC density with R_AMC > R_AS *AND* mass-loss < 90% [Msun/pc^3]",
            )

    PDF_list = np.zeros_like(R_centres)


    R_indices = np.array_split(range(len(R_centres)), MPI_size)[MPI_rank]

    for i in R_indices:
        R = R_centres[i]
        print(i, "\t - R [pc]:", R)
        if UNPERTURBED:
            weights = AMC_weights
        else:
            weights = AMC_weights_surv
            # weights = AMC_weights_AScut
        inds = weights[:, i] > 0
        # inds = np.arange(len(mass_ini_all))

        # Calculate distributions of R and M
        PDF_list[i] = calc_distributions(
            R, mass_ini_all[inds], mass_all[inds], radius_all[inds], weights[inds, i]
        )  # just pass the AMC weight at that radius


    print(R_centres)
    #R_centres is in pc
    print("Encounter rate [day^-1]:", np.trapz(PDF_list, R_centres) * 60 * 60 * 24)

    # Save the outputs
    # if not UNPERTURBED:
    out_text = PROFILE + circ_text + cut_text

    if UNPERTURBED:
        out_text += "_unperturbed"
    out_text += IDstr_out + ".txt"
    # if (UNPERTURBED):
    # _unperturbed.txt"
    # np.savetxt(output_dir + 'Rvals_distributions_' + PROFILE + '.txt', Rvals_distr)
    np.savetxt(
        dirs.data_dir + "EncounterRate_" + out_text,
        np.column_stack([R_centres, PDF_list]),
        delimiter=", ",
        header="Columns: R orbit [pc], surv_prob, MC radial distrib (dGamma/dR [pc^-1 s^-1])",
    )


# ------------------------------


def load_AMC_results(Rlist):
    Rkpc_list = Rlist / 1e3

    a_pc_all = np.array([])
    mass_ini_all = np.array([])
    mass_all = np.array([])
    radius_all = np.array([])
    e_all = np.array([])
    a_all = np.array([])

    # Divide up the processes for each MPI process
    R_vals = np.array_split(Rkpc_list, MPI_size)[MPI_rank]

    for i, Rkpc in enumerate(R_vals):
        fname = dirs.montecarlo_dir + "AMC_samples_a=%.4f_%s%s%s.txt" % (
            Rkpc*1e3,
            PROFILE,
            circ_text,
            IDstr,
        )

        columns = (
            3,
            4,
        )  # FIXME: Need to edit this if I've removed delta from the output files...
        if UNPERTURBED:
            columns = (0, 1)

        mass_ini = np.loadtxt(
            fname,
            delimiter=", ",
            dtype="f8",
            usecols=(0,),
            unpack=True,
            max_rows=max_rows,
        )
        mass, radius = np.loadtxt(
            fname,
            delimiter=", ",
            dtype="f8",
            usecols=columns,
            unpack=True,
            max_rows=max_rows,
        )
        e = np.loadtxt(
            fname,
            delimiter=", ",
            dtype="f8",
            usecols=(6,),
            unpack=True,
            max_rows=max_rows,
        )

        a_pc_all = np.concatenate((a_pc_all, np.ones_like(mass_ini) * R_vals[i] * 1e3))
        mass_ini_all = np.concatenate((mass_ini_all, mass_ini))
        mass_all = np.concatenate((mass_all, mass))
        radius_all = np.concatenate((radius_all, radius))
        e_all = np.concatenate((e_all, e))

    return mass_ini_all, mass_all, radius_all, e_all, a_pc_all


G_N = (
    6.67408e-11 * 6.7702543e-20
)  # pc^3 solar mass^-1 s^-2 (conversion: m^3 kg^-1 s^-2 to pc^3 solar mass^-1 s^-2)
# G_N = 4.302e-3


# BJK: It turns out this integral can be done analytically...
def int_P_R(r, a, e):
    x = r / a
    A = np.clip(e ** 2 - (x - 1) ** 2, 0, 1e30)

    res = (1 / np.pi) * (-np.sqrt(A) + np.arctan((x - 1) / np.sqrt(A)))
    return res


def P_R(r, a, e):
    x = r / a
    return (1 / a) * (1 / np.pi) * (2 / x - (1 - e ** 2) / x ** 2 - 1) ** -0.5


def calc_P_R(R_bin_edges, a, e):

    delta = 0
    r_min = a * (1 - e)
    r_max = a * (1 + e)

    frac = np.zeros(R_bin_edges.size - 1)

    if e < 1e-3:
        ind = np.digitize(a, R_bin_edges)
        frac[ind] = 1.0 / (R_bin_edges[ind + 1] - R_bin_edges[ind])
        return frac

    i0 = np.digitize(r_min, R_bin_edges) - 1
    i1 = np.digitize(r_max, R_bin_edges)

    # i0 = int(np.clip(i0, 0, R_bin_edges.size-1))
    # i1 = int(np.clip(i1, 0, R_bin_edges.size-1))
    # if (i0 < 0):
    #    i0 = 0

    if i1 > (len(R_bin_edges) - 1):
        i1 = len(R_bin_edges) - 1

    # print(i0, r_min, R_bin_edges[i0], R_bin_edges[i0+1])
    # print(i1, r_max, R_bin_edges[i1], R_bin_edges[i1+1])

    # for i in range(R_bin_edges.size-1):

    for i in range(i0, i1):
        # frac[i] = quad(dPdr_corrected, R_bin_edges[i], R_bin_edges[i+1], epsrel=1e-4)[0]
        R2 = np.clip(R_bin_edges[i + 1], r_min, r_max)
        R1 = np.clip(R_bin_edges[i], r_min, r_max)
        # print(R1, R2)
        if R1 < r_max and R2 > r_min:
            if R1 == r_min:
                term1 = -0.5
            else:
                term1 = int_P_R(R1, a, e)

            if R2 == r_max:
                term2 = 0.5
            else:
                term2 = int_P_R(R2, a, e)

            # Convert the integrated probability into a differential estimate
            # frac[i] = (term2 - term1)
            frac[i] = (term2 - term1) / (R_bin_edges[i + 1] - R_bin_edges[i])

    return frac


# ---------------------------


def calculate_survivalprobability(a_grid, a_all, m_final, m_ini, r_final):

    # Count number of (surviving) AMC samples for each value of a
    Nsamp_a = np.zeros(len(a_grid))
    Nsurv_a = np.zeros(len(a_grid))
    for i in range(len(a_grid)):
        Nsamp_a[i] = np.sum(np.isclose(a_all, a_grid[i], rtol=1e-5))
        Nsurv_a[i] = np.sum(np.isclose(a_all, a_grid[i], rtol=1e-5) & (m_final >= M_cut))

    # print(Nsamp_a)
    # print(Nsurv_a)

    # Here, we only need to reweight by the unperturbed mass function
    AMC_MF_unpert = mass_function.PowerLawMassFunction(m_a=in_maeV, gamma=in_gg)

    psurv_a_AScut = np.zeros(len(a_grid))
    for i in range(len(a_grid)):
        inds = np.isclose(a_all,a_grid[i], rtol=1e-5)
        AS_mask = (r_AS(m_ini[inds]) < r_final[inds]) & (m_final[inds] >= M_cut)
        p_target = AMC_MF_unpert.dPdlogM(m_ini[inds])
        p_sample = 1 / (np.log(AMC_MF_unpert.mmax) - np.log(AMC_MF_unpert.mmin))
        m_w = p_target / p_sample
        psurv_a_AScut[i] = np.sum(m_w * AS_mask) / np.sum(inds)

    return Nsurv_a / Nsamp_a, psurv_a_AScut


# ---------------------------


def calculate_weights(R_bin_edges, a_grid, a, e, mass, mass_ini, radius):

    a_bin_edges = np.sqrt(a_grid[:-1] * a_grid[1:])
    a_bin_edges = np.append(a_grid[0] / 1.5, a_bin_edges)
    a_bin_edges = np.append(a_bin_edges, a_grid[-1] * 1.5)
    delta_a = np.diff(a_bin_edges)  # Bin spacing in a

    # Count number of AMC samples for each value of a
    Nsamp_a = np.zeros(len(a_grid))
    for i in range(len(a_grid)):
        Nsamp_a[i] = np.sum(np.isclose(a,a_grid[i], rtol=1e-5))

    N_samps_tot = len(a)
    # Estimate the sampling probability of a.
    # We use a (more or less) regular (log) grid of a
    # so the probability of sampling a particular
    # value is proportional to the number of samples
    # at that particular value of a, divided by the
    # width of the bin in a.
    P_samp_a = (Nsamp_a / N_samps_tot) / delta_a
    # If we integrate this thing int P_samp_a da we get 1.
    # #ImportanceSampling

    weights = np.zeros([a.size, R_bin_edges.size - 1])
    for i in tqdm(range(a.size)):
        w = calc_P_R(R_bin_edges, a[i], e[i])

        correction = 1.0
        #print(a_grid,a[i])
        #print(np.isclose(a_grid,a[i], rtol=1e-5))
        #print(P_samp_a[np.isclose(a_grid,a[i], rtol=1e-5)])
        P = (
            4
            * np.pi
            * a[i] ** 2
            * Galaxy.rhoNFW(a[i])
            * correction
            / (P_samp_a[np.isclose(a_grid,a[i], rtol=1e-5)] * N_samps_tot)
        )
        # P = 4*np.pi*a[i]**2*NE.rhoNFW(a[i])*correction/(P_samp_a[a_grid == a[i]]*N_samps)
        weights[i, :] = w * P

    weights_survived = weights * np.atleast_2d((mass >= M_cut)).T
    
    weights_masscut = weights * np.atleast_2d((mass >= 1e-1 * mass_ini)).T

    mass_edges = np.geomspace(1e-6 * AMC_MF.mmin, 10*M0, num=Nbins_mass + 1)
    i0 = np.digitize(M0, mass_edges)
    deltam = mass_edges[i0 + 1] - mass_edges[i0]


    if (MASS_CHOICE.lower() == "full"):
        dPdM_ini = lambda x: AMC_MF.dPdlogM(x)/x
    else:
        def dPdM_ini(x):
            res = 0.0 * x
            res[np.digitize(x, mass_edges) == i0] = 1.0 / deltam
            return res


    AS_mask = (r_AS(mass_ini) < radius) & (mass >= M_cut)

    # Here, we only need to reweight by the unperturbed mass function
    # AMC_MF_unpert = mass_function.PowerLawMassFunction(m_a = in_maeV, gamma = in_gg)
    p_target = dPdM_ini(mass_ini)
    p_sample = 1 / (np.log(AMC_MF.mmax) - np.log(AMC_MF.mmin))
    m_w = p_target / p_sample
    # m_w = p_target/np.sum(p_target)
    # BJK: Need to reweight by mass function...
    weights_AScut = weights * np.atleast_2d(m_w * AS_mask).T
    weights_AScut_masscut = weights_AScut * np.atleast_2d((mass >= 1e-1 * mass_ini)).T

    return (
        weights,
        weights_survived,
        weights_masscut,
        weights_AScut,
        weights_AScut_masscut,
    )


# -----------------------------


def calculate_weights_circ(a_grid, a, e, mass, mass_ini, radius):

    a_bin_edges = np.sqrt(a_grid[:-1] * a_grid[1:])
    a_bin_edges = np.append(a_grid[0] / 1.5, a_bin_edges)
    a_bin_edges = np.append(a_bin_edges, a_grid[-1] * 1.5)
    delta_a = np.diff(a_bin_edges)  # Bin spacing in a

    # Count number of AMC samples for each value of a
    Nsamp_a = np.zeros(len(a_grid))
    for i in range(len(a_grid)):
        Nsamp_a[i] = np.sum(a == a_grid[i])

    # Estimate the sampling probability of a as 1/delta_a
    P_samp_a = 1 / delta_a
    # Then normalise to give a PDF (roughly)
    # P_samp_a /= np.sum(P_samp_a)

    weights = np.zeros([a.size, a_grid.size])
    for i in tqdm(range(a.size)):
        w = [a[i] == a_grid]

        correction = 1.0
        P = (
            4
            * np.pi
            * a[i] ** 2
            * Galaxy.rhoNFW(a[i])
            * correction
            / (Nsamp_a[a_grid == a[i]])
        )
        weights[i, :] = w * P

    weights_survived = weights * np.atleast_2d((mass >= M_cut)).T
    weights_masscut = weights * np.atleast_2d((mass >= 1e-1 * mass_ini)).T

    AS_mask = (r_AS(mass_ini) < radius) & (mass >= M_cut)

    # Here, we only need to reweight by the unperturbed mass function
    AMC_MF_unpert = mass_function.PowerLawMassFunction(m_a=in_maeV, gamma=in_gg)
    p_target = AMC_MF_unpert.dPdlogM(mass_ini)
    p_sample = 1 / (np.log(AMC_MF_unpert.mmax) - np.log(AMC_MF_unpert.mmin))
    m_w = p_target / p_sample
    # m_w = p_target/np.sum(p_target)
    # BJK: Need to reweight by mass function...
    weights_AScut = weights * np.atleast_2d(m_w * AS_mask).T

    weights_AScut_masscut = weights_AScut * np.atleast_2d((mass >= 1e-1 * mass_ini)).T

    return (
        weights,
        weights_survived,
        weights_masscut,
        weights_AScut,
        weights_AScut_masscut,
    )


# ------------------------------


def calc_distributions(R, mass_ini, mass, radius, weights_R):
    # Weights should just be a number per AMC for the weight at the particular radius R
    # This should all work the same as before but now reads in all AMCs with the associated weights
    Rkpc = R / 1e3

    rho_loc = Galaxy.rhoNFW(R)
    rho_crit = rho_loc * k

    total_weight = np.sum(weights_R)

    if total_weight > 0:
        integrand = 0
        # psurv       = N_AMC/Nini # survival probability at a given galactocentric radius # FIXME: This needs to include eccentricity
        # surv_prob   = np.append(surv_prob, psurv)

        # AMC Mass
        mass_edges = np.geomspace(1e-6 * AMC_MF.mmin, 10*M0, num=Nbins_mass + 1)

        mass_centre = np.sqrt(mass_edges[1:] * mass_edges[:-1])  # Geometric Mean

        # AMC radius
        rad_edges = np.geomspace(1e-11, 1e0, num=Nbins_radius + 1)
        rad_centre = np.sqrt(rad_edges[1:] * rad_edges[:-1])  # Geometric Mean

        rho = NE.density(mass, radius)  # NB: this is the average density

        i0 = np.digitize(M0, mass_edges)
        deltam = mass_edges[i0 + 1] - mass_edges[i0]

        
        if (MASS_CHOICE.lower() == "full"):
            dPdM_ini = lambda x: AMC_MF.dPdlogM(x)/x
        else:
            def dPdM_ini(x):
                res = 0.0 * x
                res[np.digitize(x, mass_edges) == i0] = 1.0 / deltam
                return res

        

        beta = mass / mass_ini
        
        if (total_weight < -1e5):
            plt.figure()
            plt.hist(beta, bins = np.geomspace(1e-3, 1, 10))
            plt.xscale('log')        
            plt.show()

        if UNPERTURBED:
            # beta = np.ones_like(mass)
            dPdM = dPdM_ini(mass_centre)
        else:

            dPdM = 0.0 * mass_centre
            for i, M in enumerate(mass_centre):
                Mi_temp = M / beta
                samp_list = (1 / beta) * dPdM_ini(Mi_temp) * weights_R
                # samp_list[Mi_temp < mmin] = 0
                # samp_list[Mi_temp > mmax] = 0

                if not AS_CUT:
                    dPdM[i] = np.sum(samp_list)
                else:
                    # Cut version
                    mask = rho < (k_AMC / alpha_AS) ** 3 * M ** 2 / beta
                    #print(np.sum(mask))
                    if np.sum(mask) > 0:
                        dPdM[i] = np.sum(samp_list[mask])

            #print(dPdM)
            if (np.sum(dPdM) > 0):
                dPdM /= np.trapz(dPdM, mass_centre)

            np.savetxt(
                dirs.data_dir
                + "distributions/distribution_mass_%.4f_%s%s%s%s.txt"
                % (Rkpc*1e3, PROFILE, circ_text, cut_text, IDstr_out),
                np.column_stack([mass_centre, dPdM]),
                delimiter=", ",
                header="M_f [M_sun], P(M_f) [M_sun^-1]",
            )

        dPdr = np.zeros(len(rad_centre))
        dPdr_corr = np.zeros(len(rad_centre))

        # dP(interaction)/dr = int [dP/dMdr P(interaction|M, r)] dM
        if PROFILE == "NFW":
            c = 100
            rho_AMC = (
                rho * c ** 3 / (3 * NE.f_NFW(c))
            )  # Convert mean density rhoi to AMC density
            x_cut = NE.x_of_rho(rho_crit / rho_AMC)

        elif PROFILE == "PL":
            x_cut = (rho / (4 * rho_crit)) ** (4 / 9)

        for ii, ri in enumerate(tqdm(rad_centre)):
            ri = rad_centre[ii]

            Mf_temp = (4 * np.pi / 3) * rho * ri ** 3
            Mi_temp = Mf_temp / beta

            # Integrand = dP/dM dM/dr P(beta)/beta
            samp_list = dPdM_ini(Mi_temp) / beta * (3 * Mf_temp / ri) * weights_R

            # Velocity dispersion at galactocentric radius R
            # Factor of sqrt(2) because it's the relative velocity (difference between 2 MB distributions)
            sigma_u = np.sqrt(2) * Galaxy.sigma(R) * (3.24078e-14)  # pc/s
            M_NS = 1.4
            R_cut = G_N * M_NS / sigma_u ** 2
            sigmau_corr = (
                np.sqrt(8 * np.pi)
                * sigma_u
                * ri ** 2
                * (1.0 + R_cut / ri)
                * np.minimum(x_cut ** 2, np.ones_like(ri))
            )

            if not AS_CUT:
                dPdr[ii] = np.sum(samp_list)
                dPdr_corr[ii] = np.sum(samp_list * sigmau_corr)

            else:
                mask = ri > alpha_AS * (Mf_temp / beta) ** (-1 / 3)
                if np.sum(mask) > 0:
                    dPdr[ii] = np.sum(samp_list[mask])
                    dPdr_corr[ii] = np.sum(samp_list[mask] * sigmau_corr[mask])

        n_dist = Galaxy.nNS_sph(R)  # NS distribution at R in pc^-3
        #print("N_dist:", n_dist)

        sigmau_avg = np.trapz(dPdr_corr, rad_centre)
        if (sigmau_avg > 0):
            dPdr_corr = dPdr_corr / sigmau_avg

        if (np.sum(dPdr) > 0):
            dPdr = dPdr / np.trapz(dPdr, rad_centre)

        # dGamma/dr_GC
        integrand = n_dist * sigmau_avg / M0  # rho_NFW

        # rho_NFW is now applied in calculate_weights

        outfile_text = ""
        if UNPERTURBED:
            outfile_text = PROFILE + circ_text + cut_text + "_unperturbed"
        else:
            outfile_text = "%.4f_%s%s%s" % (Rkpc*1e3, PROFILE, circ_text, cut_text)

        np.savetxt(
            dirs.data_dir
            + "distributions/distribution_radius_"
            + outfile_text
            + IDstr_out
            + ".txt",
            np.column_stack([rad_centre, dPdr, dPdr_corr]),
            delimiter=", ",
            header="Columns: R_MC [pc], P(R_MC) [1/pc], Cross-section weighted P(R_MC) [1/pc]",
        )

        return integrand
    else:
        return 0


# ----------------------

main()

if MPI_rank == 0:
    print("----->Done.")
