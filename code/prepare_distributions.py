#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
# from scipy.integrate import quad
# from scipy import interpolate
# import AMC
import mass_function
import NSencounter as NE
import perturbations as PB
import tools
from tools import r_AS
import glob

try:
    from tqdm import tqdm
except ImportError as err:
    tqdm = lambda x: x

import MilkyWay as Galaxy
import Andromeda as Galaxy

import argparse
import sys
import os
import re
import warnings

import params

import dirs

G_N = (
    6.67408e-11 * 6.7702543e-20
)  # pc^3 solar mass^-1 s^-2 (conversion: m^3 kg^-1 s^-2 to pc^3 solar mass^-1 s^-2)
# G_N = 4.302e-3

#Make distributions
if not os.path.exists(dirs.data_dir + "distributions/"):
    os.makedirs(dirs.data_dir + "distributions/")

warnings.filterwarnings("error")

#Some internal parameters relevant for prepare_distributions
M_cut = 1e-29
Nbins_mass = 300
Nbins_radius = 300  # Previously 500



k_AMC = (3 / (4 * np.pi)) ** (1 / 3)


def prepare_distributions(m_a, profile,  mass_function_ID, galaxyID = "MW", circular=False, unperturbed=False,max_rows=None,  IDstr=""):
        
    file_suffix = tools.generate_suffix(profile, mass_function_ID, circular=circular,
                                        AScut=False, unperturbed=unperturbed, IDstr=IDstr, verbose=True)
                                        
    file_suffix_AScut = tools.generate_suffix(profile, mass_function_ID, circular=circular,
                                        AScut=True, unperturbed=unperturbed, IDstr=IDstr, verbose=True)
    
    #Set up the mass function
    AMC_MF, M0 = mass_function.get_mass_function(mass_function_ID, m_a, profile, unperturbed=unperturbed, Nbins_mass=Nbins_mass)
    
    # Gather the list of files to be used, then loop over semi-major axis a
    a_grid = None
    f_search = dirs.montecarlo_dir + "AMC_samples_*" + file_suffix + ".txt"
    files = glob.glob(f_search)

    a_grid = np.zeros(len(files))

    for i, fname in enumerate(files):
        m = re.search("AMC_samples_a=(.+?)_" + file_suffix + ".txt", fname)
        if m:
            a_string = m.group(1)
        a_grid[i] = float(a_string) # conversion to pc

    a_grid = np.sort(a_grid)
    

    # Edges to use for the output bins in R (galactocentric radius, pc)
    if circular:
        R_centres = 1.0 * a_grid
    else:
        R_bin_edges = np.geomspace(1e-2, 60e3, 101)
        R_centres = np.sqrt(R_bin_edges[:-1] * R_bin_edges[1:])

    mass_ini_all, mass_all, radius_all, e_all, a_all = load_AMC_results(a_grid, file_suffix, unperturbed, max_rows)

    # ----------------------------


    # Re-weight the samples according to radius
    if circular:
        (
            AMC_weights,
            AMC_weights_surv,
            AMC_weights_masscut,
            AMC_weights_AScut,
            AMC_weights_AScut_masscut,
        ) = calculate_weights(
            a_grid, a_all, e_all, mass_all, mass_ini_all, radius_all, Galaxy, AMC_MF, circular=True
        )
    else:
        (
            AMC_weights,
            AMC_weights_surv,
            AMC_weights_masscut,
            AMC_weights_AScut,
            AMC_weights_AScut_masscut,
        ) = calculate_weights(
            R_bin_edges, a_grid, a_all, e_all, mass_all, mass_ini_all, radius_all, Galaxy, AMC_MF
        )  # Just pass the eccentricities and semi major axes


    print("> Calculating survival probabilities...")
    #BJK: I'm not sure we need a separate function to calculate the survival probability
    #BJK: I think this can be done based on the weights...
    # Calculate the survival probability as a function of a
    psurv_a_list, psurv_a_AScut_list = calculate_survivalprobability(
        a_grid, a_all, mass_all, mass_ini_all, radius_all, AMC_MF
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
    if not unperturbed:
        # np.savetxt(output_dir + 'Rvals_distributions_' + PROFILE + '.txt', Rvals_distr)
        if not circular:
            np.savetxt(
                dirs.data_dir + "SurvivalProbability_a_" + file_suffix + ".txt",
                np.column_stack([a_grid, psurv_a_list, psurv_a_AScut_list]),
                delimiter=", ",
                header="Columns: semi-major axis [pc], survival probability, survival probability for AMCs passing the AS cut",
            )
        np.savetxt(
            dirs.data_dir
            + f"SurvivalProbability_R_{file_suffix}.txt",
            np.column_stack(
                [
                    R_centres,
                    psurv_R_list,
                    P_r_weights_AScut/P_r_weights,
                    #P_r_weights_surv,
                    #P_r_weights_masscut,
                    #P_r_weights_AScut,
                    #P_r_weights_AScut_masscut,
                ]
            ),
            delimiter=", ",
            #header="Columns: galactocentric radius [pc], survival probability, Initial AMC density [Msun/pc^3], Surviving AMC density [Msun/pc^3], Surviving AMC density with mass-loss < 90% [Msun/pc^3], Surviving AMC density with R_AMC > R_AS [Msun/pc^3], Surviving AMC density with R_AMC > R_AS *AND* mass-loss < 90% [Msun/pc^3]",
            header="Columns: galactocentric radius [pc], survival probability, survival probability (with AS cut)",
        )

    PDF_list = np.zeros_like(R_centres)
    PDF_list_AScut = np.zeros_like(R_centres)


    
    for i, R in enumerate(tqdm(R_centres, desc = "Calculating distributions")):
        R = R_centres[i]
        #print(i, "\t - R [pc]:", R)
        if unperturbed:
            weights = AMC_weights
        else:
            weights = AMC_weights_surv
            # weights = AMC_weights_AScut
        inds = weights[:, i] > 0
        # inds = np.arange(len(mass_ini_all))

        # Calculate distributions of R and M
        PDF_list[i] = calc_distributions(
            R, mass_ini_all[inds], mass_all[inds], radius_all[inds], weights[inds, i], Galaxy, AMC_MF, unperturbed, False, profile, file_suffix
        )  # just pass the AMC weight at that radius
        
        # Calculate distributions of R and M
        PDF_list_AScut[i] = calc_distributions(
            R, mass_ini_all[inds], mass_all[inds], radius_all[inds], weights[inds, i], Galaxy, AMC_MF, unperturbed, True, profile, file_suffix_AScut
        )  # just pass the AMC weight at that radius

    #R_centres is in pc
    print("Encounter rate (without AScut)[day^-1]:\t", np.trapz(PDF_list, R_centres) * 60 * 60 * 24)
    print("Encounter rate (including AScut) [day^-1]:\t", np.trapz(PDF_list_AScut, R_centres) * 60 * 60 * 24)

    np.savetxt(
        dirs.data_dir + "EncounterRate_" + file_suffix + ".txt",
        np.column_stack([R_centres, PDF_list, PDF_list_AScut]),
        delimiter=", ",
        header="Columns: R orbit [pc], surv_prob, Encounter radial distrib (dGamma/dR [pc^-1 s^-1])",
    )


# ------------------------------


def load_AMC_results(alist, file_suffix, unperturbed, max_rows):
    a_vals = alist / 1e3

    a_pc_all = np.array([])
    mass_ini_all = np.array([])
    mass_all = np.array([])
    radius_all = np.array([])
    e_all = np.array([])
    a_all = np.array([])

    for i, a_kpc in enumerate(tqdm(a_vals, desc="Loading Monte Carlo simulations")):
        fname = dirs.montecarlo_dir + f"AMC_samples_a={a_kpc*1e3:.4f}_{file_suffix}.txt"

        columns = (3, 4,) 
        if unperturbed:
            columns = (0, 1)

        mass_ini = np.loadtxt( fname, delimiter=", ", dtype="f8", usecols=(0,), unpack=True, max_rows=max_rows, )
        mass, radius = np.loadtxt( fname, delimiter=", ", dtype="f8", usecols=columns, unpack=True, max_rows=max_rows, )
        e = np.loadtxt( fname, delimiter=", ", dtype="f8", usecols=(6,), unpack=True, max_rows=max_rows, )

        a_pc_all = np.concatenate((a_pc_all, np.ones_like(mass_ini) * a_kpc * 1e3))
        mass_ini_all = np.concatenate((mass_ini_all, mass_ini))
        mass_all = np.concatenate((mass_all, mass))
        radius_all = np.concatenate((radius_all, radius))
        e_all = np.concatenate((e_all, e))

    return mass_ini_all, mass_all, radius_all, e_all, a_pc_all





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

#---------------------------------

def calculate_weights(R_bin_edges, a_grid, a, e, mass, mass_ini, radius, Galaxy, AMC_MF, circular=False):

    a_bin_edges = np.sqrt(a_grid[:-1] * a_grid[1:])
    a_bin_edges = np.append(a_grid[0] / 1.5, a_bin_edges)
    a_bin_edges = np.append(a_bin_edges, a_grid[-1] * 1.5)
    delta_a = np.diff(a_bin_edges)  # Bin spacing in a

    # Count number of AMC samples for each value of a
    Nsamp_a = np.zeros(len(a_grid))
    for i in range(len(a_grid)):
        #Number of samples at semi-major axis a
        Nsamp_a[i] = np.sum(np.isclose(a,a_grid[i], rtol=1e-5))

    if (circular):
        # Estimate the sampling probability of a as 1/delta_a
        P_samp_a = 1 / delta_a
        # Then normalise to give a PDF (roughly)
        # P_samp_a /= np.sum(P_samp_a)
    else:
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
    for i in tqdm(range(a.size), desc="Calculating weights"):
        a_ind = np.isclose(a_grid,a[i], rtol=1e-5)
        if (circular):
            w = [a[i] == a_grid]
            P_s = Nsamp_a[a_ind]
        else:
            w = calc_P_R(R_bin_edges, a[i], e[i])
            P_s = P_samp_a[a_ind] * N_samps_tot

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
            / P_s
        )
        # P = 4*np.pi*a[i]**2*NE.rhoNFW(a[i])*correction/(P_samp_a[a_grid == a[i]]*N_samps)
        weights[i, :] = w * P

        weights_survived = weights * np.atleast_2d((mass >= M_cut)).T
        
    if (AMC_MF.type == "extended"):
        weights_masscut = weights * np.atleast_2d((mass >= 1e-1 * mass_ini)).T

        dPdM_ini = lambda x: AMC_MF.dPdlogM(x)/x

        AS_mask = (r_AS(mass_ini, AMC_MF.m_a) < radius) & (mass >= M_cut)

        # Here, we only need to reweight by the unperturbed mass function
        # AMC_MF_unpert = mass_function.PowerLawMassFunction(m_a = in_maeV, gamma = in_gg)
        p_target = dPdM_ini(mass_ini)
        p_sample = 1 / (np.log(AMC_MF.mmax) - np.log(AMC_MF.mmin))
        m_w = p_target / p_sample
        # m_w = p_target/np.sum(p_target)
        
        weights_AScut = weights * np.atleast_2d(m_w * AS_mask).T
        weights_AScut_masscut = weights_AScut * np.atleast_2d((mass >= 1e-1 * mass_ini)).T
        
    elif (AMC_MF.type == "delta"):
        
        beta = mass/mass_ini
        m_final_corr = AMC_MF.M0*beta
        
        weights_masscut = weights * np.atleast_2d((beta >= 1e-1)).T
        
        r_final_corr = radius*(m_final_corr/mass)**(1/3)
        AS_mask = (r_AS(AMC_MF.M0, AMC_MF.m_a) < r_final_corr) & (m_final_corr >= M_cut)
        m_w = np.ones_like(AS_mask)
        weights_AScut = weights * np.atleast_2d(m_w * AS_mask).T
        weights_AScut_masscut = weights_AScut * np.atleast_2d((beta >= 1e-1)).T


    return (
        weights,
        weights_survived,
        weights_masscut,
        weights_AScut,
        weights_AScut_masscut,
    )


# ---------------------------

#BJK: I'm here...
def calculate_survivalprobability(a_grid, a_all, m_final, m_ini, r_final, AMC_MF):

    # Count number of (surviving) AMC samples for each value of a
    Nsamp_a = np.zeros(len(a_grid))
    Nsurv_a = np.zeros(len(a_grid))
    for i in range(len(a_grid)):
        Nsamp_a[i] = np.sum(np.isclose(a_all, a_grid[i], rtol=1e-5))
        Nsurv_a[i] = np.sum(np.isclose(a_all, a_grid[i], rtol=1e-5) & (m_final >= M_cut))

    psurv_a_AScut = np.zeros(len(a_grid))
    for i in range(len(a_grid)):
        inds = np.isclose(a_all,a_grid[i], rtol=1e-5)
        
        if (AMC_MF.type == "extended"):
            #For an extended distribution, we need to reweight the masses,
            #because for the MC they were sampled log-flat
            AS_mask = (r_AS(m_ini[inds], AMC_MF.m_a) < r_final[inds]) & (m_final[inds] >= M_cut)
            p_target = AMC_MF.dPdlogM(m_ini[inds])
            p_sample = 1 / (np.log(AMC_MF.mmax) - np.log(AMC_MF.mmin))
            m_w = p_target / p_sample
            
        elif (AMC_MF.type == "delta"):
            #For a delta-function, we ignore the sampled mass function, calculate the 
            #the mass-loss (beta) and then calculate the distribution of masses
            #and radii starting from a single initial mass (AMC_MF.M0)
            beta = m_final[inds]/m_ini[inds]
            m_final_corr = AMC_MF.M0*beta
            r_final_corr = r_final[inds]*(m_final_corr/m_final[inds])**(1/3)
            AS_mask = (r_AS(AMC_MF.M0, AMC_MF.m_a) < r_final_corr) & (m_final_corr >= M_cut)
            m_w = np.ones_like(AS_mask)
            
        psurv_a_AScut[i] = np.sum(m_w * AS_mask) / np.sum(inds)

    return Nsurv_a / Nsamp_a, psurv_a_AScut


# ------------------------------


def calc_distributions(R, mass_ini, mass, radius, weights_R, Galaxy, AMC_MF, unperturbed, AScut, profile, file_suffix):
    # Weights should just be a number per AMC for the weight at the particular radius R
    # This should all work the same as before but now reads in all AMCs with the associated weights
    Rkpc = R / 1e3

    rho_loc = Galaxy.rhoNFW(R)
    rho_crit = rho_loc * params.min_enhancement

    total_weight = np.sum(weights_R)

    if total_weight > 0:
        integrand = 0
        # psurv       = N_AMC/Nini # survival probability at a given galactocentric radius # FIXME: This needs to include eccentricity
        # surv_prob   = np.append(surv_prob, psurv)

        # AMC Mass
        mass_edges = AMC_MF.mass_edges

        mass_centre = np.sqrt(mass_edges[1:] * mass_edges[:-1])  # Geometric Mean

        # AMC radius
        rad_edges = np.geomspace(1e-11, 1e0, num=Nbins_radius + 1)
        rad_centre = np.sqrt(rad_edges[1:] * rad_edges[:-1])  # Geometric Mean

        rho = NE.density(mass, radius)  # NB: this is the average density
                
        dPdM_ini = lambda x: AMC_MF.dPdlogM(x)/x  

        beta = mass / mass_ini
        
        if (total_weight < -1e5):
            plt.figure()
            plt.hist(beta, bins = np.geomspace(1e-3, 1, 10))
            plt.xscale('log')        
            plt.show()

        if unperturbed:
            # beta = np.ones_like(mass)
            dPdM = dPdM_ini(mass_centre)
        else:

            dPdM = 0.0 * mass_centre
            for i, M in enumerate(mass_centre):
                Mi_temp = M / beta
                samp_list = (1 / beta) * dPdM_ini(Mi_temp) * weights_R
                # samp_list[Mi_temp < mmin] = 0
                # samp_list[Mi_temp > mmax] = 0
                
                if not AScut:
                    dPdM[i] = np.sum(samp_list)
                else:
                    if (AMC_MF.type == "extended"):
                        # Cut version
                        alpha_AS = r_AS(1.0, AMC_MF.m_a)
                        mask = rho < (k_AMC / alpha_AS) ** 3 * M ** 2 / beta
                    elif (AMC_MF.type == "delta"):
                        r_f = k_AMC*(beta*AMC_MF.M0/rho)**(1/3)
                        r_AS0 = r_AS(AMC_MF.M0, AMC_MF.m_a)
                        mask = r_f > r_AS0
                        
                    #print(np.sum(mask))
                    if np.sum(mask) > 0:
                        dPdM[i] = np.sum(samp_list[mask])

            #print(dPdM)
            if (np.sum(dPdM) > 0):
                dPdM /= np.trapz(dPdM, mass_centre)

            # Test plot!!!
            
            """
            plt.figure()
            
            plt.loglog(mass_centre, dPdM)
            plt.axvline(AMC_MF.M0, linestyle='--', color='k')
            plt.axvline(AMC_MF.mmin, linestyle=':', color='k')
            plt.axvline(AMC_MF.mmax, linestyle=':', color='k')
            plt.show()
            """
            
            np.savetxt(
                dirs.data_dir
                + f"distributions/distribution_mass_{Rkpc*1e3:.4f}_{file_suffix}.txt",
                np.column_stack([mass_centre, dPdM]),
                delimiter=", ",
                header="M_f [M_sun], P(M_f) [M_sun^-1]",
            )

        dPdr = np.zeros(len(rad_centre))
        dPdr_corr = np.zeros(len(rad_centre))

        # dP(interaction)/dr = int [dP/dMdr P(interaction|M, r)] dM
        if profile == "NFW":
            c = 100
            rho_AMC = (
                rho * c ** 3 / (3 * NE.f_NFW(c))
            )  # Convert mean density rhoi to AMC density
            x_cut = NE.x_of_rho(rho_crit / rho_AMC)

        elif profile == "PL":
            x_cut = (rho / (4 * rho_crit)) ** (4 / 9)

        for i, ri in enumerate(rad_centre):
            r = rad_centre[i]

            Mf_temp = (4 * np.pi / 3) * rho * r ** 3
            Mi_temp = Mf_temp / beta

            # Integrand = dP/dM dM/dr P(beta)/beta
            samp_list = dPdM_ini(Mi_temp) / beta * (3 * Mf_temp / r) * weights_R

            #BJK: This is where I need to fix the cross-section calculation
            # Velocity dispersion at galactocentric radius R
            # Factor of sqrt(2) because it's the relative velocity (difference between 2 MB distributions)
            sigma_u = np.sqrt(2) * Galaxy.sigma(R) * (3.24078e-14)  # pc/s
            M_NS = 1.4
            R_cut = G_N * M_NS / sigma_u ** 2
            sigmau_corr = (
                np.sqrt(8 * np.pi)
                * sigma_u
                * ri ** 2
                * (1.0 + R_cut / r)
                * np.minimum(x_cut ** 2, np.ones_like(r))
            )

            if not AScut:
                dPdr[i] = np.sum(samp_list)
                dPdr_corr[i] = np.sum(samp_list * sigmau_corr)

            else:
                if (AMC_MF.type == "extended"):
                    alpha_AS = r_AS(1.0, AMC_MF.m_a)
                    mask = r > alpha_AS * (Mf_temp / beta) ** (-1 / 3)
                elif (AMC_MF.type == "delta"):
                    r_AS0 = r_AS(AMC_MF.M0, AMC_MF.m_a)
                    mask = r > r_AS0
                    
                if np.sum(mask) > 0:
                    dPdr[i] = np.sum(samp_list[mask])
                    dPdr_corr[i] = np.sum(samp_list[mask] * sigmau_corr[mask])

        n_dist = Galaxy.nNS_sph(R)  # NS distribution at R in pc^-3
        #print("N_dist:", n_dist)

        sigmau_avg = np.trapz(dPdr_corr, rad_centre)
        if (sigmau_avg > 0):
            dPdr_corr = dPdr_corr / sigmau_avg

        if (np.sum(dPdr) > 0):
            dPdr = dPdr / np.trapz(dPdr, rad_centre)

        # dGamma/dr_GC
        integrand = n_dist * sigmau_avg / AMC_MF.mavg  # rho_NFW


        np.savetxt(
            dirs.data_dir
            + f"distributions/distribution_radius_{Rkpc*1e3:.4f}_{file_suffix}.txt",
            np.column_stack([rad_centre, dPdr, dPdr_corr]),
            delimiter=", ",
            header="Columns: R_MC [pc], P(R_MC) [1/pc], Cross-section weighted P(R_MC) [1/pc]",
        )

        return integrand
    else:
        return 0


# ----------------------

def getOptions(args=sys.argv[1:]):
    # Parse the arguments!
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument("-profile", "--profile", help="Density profile for AMCs - `NFW` or `PL`", type=str, default="PL")
    parser.add_argument("-m_a", "--m_a", type=float, help="Axion mass in eV", default = 50e-6)
    parser.add_argument("-unperturbed", "--unperturbed", help="Calculate for unperturbed profiles?", type=bool, default=False)
    parser.add_argument("-max_rows", "--max_rows", help="Maximum number of rows to read from each file?", type=int, default=None)
    parser.add_argument("-galaxyID", "--galaxyID", type=str, help="ID of galaxy - 'MW' or 'M31'", default="MW")
    parser.add_argument("-circ", "--circular", dest="circular", action="store_true", help="Use the circular flag to force e = 0 for all orbits.")
    parser.add_argument("-MF_ID", "--mass_function_ID", help="...", type=str, default="delta_c")
    parser.add_argument("-IDstr", "--IDstr", type=str, help = "ID string to label the output files.", default="")
    parser.set_defaults(circular=False)
    parser.set_defaults(AScut=False)

    options = parser.parse_args(args)
    return options
    
    
# ----------------------
    
if __name__ == '__main__':
    opts = getOptions(sys.argv[1:])
    
    prepare_distributions(opts.m_a, opts.profile, opts.MF_ID,  opts.galaxyID,  opts.circular, opts.unperturbed, opts.max_rows, opts.IDstr)


