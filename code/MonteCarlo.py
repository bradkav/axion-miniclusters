#!/usr/bin/env python3

import numpy as np
#import matplotlib.pyplot as plt
import AMC
import perturbations as PB
import mass_function
import orbits



try:
    from tqdm import tqdm
except ImportError as err:
    def tqdm(x):
        return x
        
import argparse
import sys
import tools

import dirs
import params

import MilkyWay
import Andromeda

from matplotlib import pyplot as plt

SAVE_OUTPUT = True
VERBOSE = False


#This script can also be run stand-alone, with command-line arguments defined at the bottom of this file, using the `getOptions` function
def Run_AMC_MonteCarlo(a0, N_AMC, m_a, profile, AMC_MF, galaxyID = "MW", circular=False, IDstr=""):

    if (galaxyID == "MW"):
        Galaxy = MilkyWay
    elif (galaxyID == "M31"):
        Galaxy = Andromeda
    else:
        raise ValueError("Invalid galaxyID.")

    M0 = AMC_MF.M0

    # Perturber parameters
    Mp = 1.0*Galaxy.M_star_avg
    a0 *= 1e3 #Get semi-major axis in parsec
 
    # Here we calculate the number of AMCs in the
    # We need to sample the number of AMCs in bins of a given radius
    Tage = 4.26e17

    sig_rel = np.sqrt(2)*Galaxy.sigma(a0)
    #print(sig_rel)

    R_bins = np.linspace(0.1*1e3,100*1e3,10)
    R_mid = R_bins[:-1] + np.diff(R_bins)/2

    M_list_initial = []
    R_list_initial = []
    rho_list_initial = []

    M_list_final = []
    R_list_final = []
    Rloc_list_final = []
    rho_list_final = []

    N_disrupt = 0
    
    # ---------------------------------------------------------------- 
    M_list, rho_list = AMC_MF.sample_AMCs_logflat(n_samples = N_AMC)
    
    
    
    a_list = np.ones(N_AMC)*a0 

    # print(circular)
    if circular:
        e_list = np.zeros(N_AMC)
    if not circular:
        e_list = PB.sample_ecc(N_AMC)

    # Check that eccentricities are sampled correctly

    Ntotal_list = []
    Nenc_list = []

    psi_list = np.random.uniform(-np.pi/2.,np.pi/2.,size = N_AMC)

    R_i_min = 1e30
    R_i_max = -1e30
    #for j in tqdm(range(N_AMC)):
    for j in range(N_AMC):
        if (VERBOSE):
            print(j)
        #print(j)
        # Initialise the AMC
        minicluster = AMC.AMC(M = M_list[j], rho = rho_list[j], profile=profile)
        
        if (minicluster.R > R_i_max):
            R_i_max = minicluster.R
        if (minicluster.R < R_i_min):
            R_i_min = minicluster.R
        
        M_list_initial.append(minicluster.M)
        R_list_initial.append(minicluster.R)
        rho_list_initial.append(minicluster.rho)
        
    
        # Galactic parameters
        E_test = PB.Elist(sig_rel, 1.0, Mp, minicluster.M, Rrms2 = minicluster.Rrms2())
        #E_test_NFW = PB.Elist(sig_rel, 1.0, Mp, minicluster_NFW.M, Rrms2 = minicluster_NFW.Rrms2())

        #Calculate b_max based on a 'test' impact at b = 1 pc
        N_cut = int(1e5)
        bmax = ((E_test/minicluster.Ebind)*N_cut)**(1./4)
        rho0 = minicluster.rho_mean()

    
        orb = orbits.elliptic_orbit(a_list[j], e_list[j], galaxy = galaxyID)
    
        #Calculate total number of encounters
        Ntotal = min(int(N_cut),int(PB.Ntotal_ecc(Tage, bmax, orb, psi_list[j], galaxy=Galaxy, b0=0.0))) # This needs to be checked
        #print(Ntotal)
        Ntotal_list.append(Ntotal)

    
        #print(j, bmax, Ntotal)
        # Added condition to skip if no encounters
        #print(Ntotal)
        if Ntotal == 0:
            M_list_final.append(minicluster.M)
            R_list_final.append(minicluster.R)
            rho_list_final.append(minicluster.rho)
            continue

        #BJK: Deal with this!?
        #print(Ntotal, N_cut)
        if ((Ntotal == N_cut) and (profile == "PL")):
            M_list_final.append(1e-30)
            R_list_final.append(1e-30)
            rho_list_final.append(1e-30)
            continue
    
    
        Nextra = 1
        #Sample properties of the stellar encounters
        blist = PB.dPdb(bmax, Nsamples=Nextra*Ntotal)
        # print(a_list[j], e_list[j], psi_list[j])
        v_amc_list, r_interaction_list = PB.dPdVamc(orb, psi_list[j], bmax, Nsamples=Nextra*Ntotal, galaxy=Galaxy)
        Vlist = PB.dPdV(v_amc_list, Galaxy.sigma(r_interaction_list), Nsamples=Nextra*Ntotal) 
        #print(Vlist)

        Vlist = np.array(Vlist)
        M_cut = 1e-25
    
        Mlist = np.zeros(Ntotal)
        rholist = np.zeros(Ntotal)
        Etotlist = np.zeros(Ntotal)
        dElist = np.zeros(Ntotal)
        Rlist = np.zeros(Ntotal)
    
        #How long does the simulation have left?
        N_enc = 0
        T_remain = 1.0*Tage
        dT = T_remain/Ntotal
        #N_remain = N_total
        i = 0

        #Iteratively perturb the AMCs
        while T_remain > dT:
        #for i in range(Ntotal):
            T_remain -= dT
            N_enc += 1
        
            #print(minicluster.M, minicluster.R, minicluster.rho)
            if (minicluster.M < M_cut):
                #print("    Disrupted after ", N_enc, " encounters")
                minicluster.disrupt()
                #N_disrupt += 1 
                break
            else:
                E_pert = PB.Elist(Vlist[i], blist[i], Mp, minicluster.M, Rrms2 = minicluster.Rrms2()) #Previously Rrms2 = minicluster.R**2/3 for isothermal sphere
                #print(E_pert/minicluster.Ebind())
                delta_E = E_pert/minicluster.Ebind
                #dElist[i] = delta_E
                minicluster.perturb(E_pert)
            
                if (minicluster.M > M_cut):
                    #If the density of the AMC increases, recompute the number of encounters required
                    if (minicluster.rho_mean() > rho0):
                        #print(dT, T_remain)
                        E_test = PB.Elist(sig_rel, 1.0, Mp, minicluster.M, Rrms2 = minicluster.Rrms2())
                        bmax_new = ((E_test/minicluster.Ebind)*N_cut)**(1./4)
                    
                    
                        N_remain = min(int(N_cut),int(PB.Ntotal_ecc(T_remain, bmax_new, orb, psi_list[j], galaxy=Galaxy, b0=0.0)))
                        #N_remain = PB.Ntotal_ecc(T_remain, bmax, orb, psi_list[j], b0=0.0)
                        #print(T_remain, bmax, PB.Ntotal_ecc(T_remain, bmax, orb, psi_list[j], b0=0.0), N_remain)
                    
                        if (N_remain == 0):
                            break
                        else:
                            dT = T_remain/N_remain
                    
                        rho0 = minicluster.rho_mean()
                        #blist[(i+1):] = PB.dPdb(bmax, Nsamples=len(blist[(i+1):]))
                        #print(bmax_new/bmax, N_remain, dT, T_remain/Tage)
                        blist[(i+1):] = blist[(i+1):]*(bmax_new/bmax)
                        bmax = bmax_new
                i += 1         
        #print(j, Ntotal, N_enc)         
                    
            
        Nenc_list.append(N_enc)
        M_list_final.append(minicluster.M)
        R_list_final.append(minicluster.R)
        rho_list_final.append(minicluster.rho)
        #
    #print("Number of disrupted AMCs:", N_disrupt)
    M_list_initial = np.array(M_list_initial)
    R_list_initial = np.array(R_list_initial)
    rho_list_initial = np.array(rho_list_initial)

    M_list_final = np.array(M_list_final)
    R_list_final = np.array(R_list_final)
    rho_list_final = np.array(rho_list_final)
    Ntotal_list = np.array(Ntotal_list)

    #------------------------------------ Testing
    #print(M_list_initial)
    #print(Nenc_list)
    #print(M_list_final/M_list_initial)
    if (VERBOSE):
        print("p_surv = ", np.sum(M_list_final > 1e-29)/len(M_list_initial))
        print("p_surv (M_f > 10% M_i) = ", np.sum(M_list_final/M_list_initial > 1e-1)/len(M_list_initial))

    file_suffix = tools.generate_suffix(profile, AMC_MF, circular=circular, IDstr=IDstr, verbose=False)

    Results = np.column_stack([M_list_initial, R_list_initial, rho_list_initial, M_list_final, R_list_final, rho_list_final, e_list, psi_list])
    if (SAVE_OUTPUT):
        np.savetxt(dirs.montecarlo_dir + 'AMC_samples_a=%.4f_%s.txt'% (a0, file_suffix), Results, delimiter=', ', header="Columns: M initial [Msun], R initial [pc], Initial mean density rho [Msun/pc^3],  M final [Msun], R final [pc], Final mean density rho [Msun/pc^3], eccentricity, psi [rad]")
        np.savetxt(dirs.montecarlo_dir + 'AMC_Ninteractions_a=%.4f_%s.txt'% (a0, file_suffix), Ntotal_list)
        np.savetxt(dirs.montecarlo_dir + 'AMC_Ninteractions_true_a=%.4f_%s.txt'% (a0, file_suffix), Nenc_list)

    #print("R_range:", R_i_min, R_i_max)

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-a", "--semi_major_axis", type=float, help="Galactocentric semi-major axis [kpc].", required = True)
    parser.add_argument("-N", "--AMC_number", type=int, help="Number of AMCs in the simulation.", default = 10000)
    parser.add_argument("-profile", "--profile", type=str, help="Density profile - `PL` or `NFW`", default="PL")
    parser.add_argument("-galaxyID", "--galaxyID", type=str, help="ID of galaxy - 'MW' or 'M31'", default="MW")
    parser.add_argument("-m_a", "--m_a", type=float, help="Axion mass in eV", default = 50e-6)
    parser.add_argument("-MF_ID", "--mass_function_ID", help="...", type=str, default="delta_c")
    parser.add_argument("-circ", "--circular", dest="circular", action='store_true', help="Use the circular flag to force e = 0 for all orbits.")
    parser.add_argument("-IDstr", "--IDstr", type=str, help = "ID string to label the output files.", default="")
    parser.set_defaults(circular=False)
    
    options = parser.parse_args(args)
    return options


if __name__ == '__main__':
    opts = getOptions(sys.argv[1:])
    
    #Create a mass function based on the input "mass function ID"
    AMC_MF = get_mass_function(opts.MF_ID, opts.m_a, opts.profile)
    AMC_MF.label = opts.MF_ID
    
    Run_AMC_MonteCarlo(opts.semi_major_axis, opts.AMC_number, opts.m_a, opts.profile, AMC_MF, opts.galaxyID, opts.circular, opts.IDstr)