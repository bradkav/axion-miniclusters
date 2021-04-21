#!/usr/bin/env python3

import numpy as np
#import matplotlib.pyplot as plt
import AMC
import MilkyWay as MW
import perturbations as PB
import orbits
from tqdm import tqdm
import argparse
import sys

import dirs

from matplotlib import pyplot as plt

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-a", "--semi_major_axis", type=float, help="Galactocentric semi-major axis [kpc].", required = True)
    parser.add_argument("-N", "--AMC_number", type=int, help="Number of AMCs in the simulation.", default = 10000)
    parser.add_argument("-profile", "--profile", type=str, help="Density profile - `PL` or `NFW`", default="PL")
    #parser.add_argument("-circ", "--circular", type=, help="Cirular orbits or eccentric", default=False)
    
    parser.add_argument("-circ", "--circular", dest="circular", action='store_true', help="Use the circular flag to force e = 0 for all orbits.")
    parser.set_defaults(circular=False)
    
    options = parser.parse_args(args)
    return options

options = getOptions(sys.argv[1:])

SAVE_OUTPUT = True
VERBOSE = False

a0 = options.semi_major_axis*1e3           # semi-major axis, conversion to pc  
N_AMC = options.AMC_number
profile = options.profile
circular = options.circular

# Perturber parameters
Mp = 1.0*MW.M_star_avg
 
# Here we calculate the number of AMCs in the
# We need to sample the number of AMCs in bins of a given radius
Tage = 4.26e17

sig_rel = np.sqrt(2)*PB.sigma(a0)
#print(sig_rel)

#sig_rel = 10 # random test number TO BE CHECKED
R_bins = np.linspace(0.1*1e3,100*1e3,10)
R_mid = R_bins[:-1] + np.diff(R_bins)/2

M_list_initial = []
R_list_initial = []
delta_list_initial = []

M_list_final = []
R_list_final = []
Rloc_list_final = []
delta_list_final = []

N_disrupt = 0

# These are all the intrinsic parameters of the AMC
# ---------------------------------------------------------------- 
M_list, delta_list = PB.sample_AMCs_logflat(m_a = 2e-5, n_samples = N_AMC) 
# ---------------------------------------------------------------- 
# here we need the parameters of the orbits a and e
a_list = np.ones(N_AMC)*a0 

# print(circular)
if circular:
    e_list = np.zeros(N_AMC)
if not circular:
    e_list = PB.sample_ecc(N_AMC)

# Check that eccentricities are sampled correctly

Ntotal_list = []
Nenc_list = []

psi_list = np.random.uniform(-np.pi/2.,np.pi/2.,size = N_AMC) #FIXME: Should this be corrected for in signal calculation




#for j in tqdm(range(N_AMC)):
for j in range(N_AMC):
    if (VERBOSE):
        print(j)
    #print(j)
    # Initialise the AMC
    minicluster = AMC.AMC(M = M_list[j], delta = delta_list[j], profile=profile)
    M_list_initial.append(minicluster.M)
    R_list_initial.append(minicluster.R)
    delta_list_initial.append(minicluster.delta)
    
    # Galactic parameters
    E_test = PB.Elist(sig_rel, 1.0, Mp, minicluster.M, Rrms2 = minicluster.Rrms2())
    #E_test_NFW = PB.Elist(sig_rel, 1.0, Mp, minicluster_NFW.M, Rrms2 = minicluster_NFW.Rrms2())

    #Calculate b_max based on a 'test' impact at b = 1 pc
    N_cut = 1e6
    bmax = ((E_test/minicluster.Ebind())*N_cut)**(1./4)
    rho0 = minicluster.rho_mean()

    
    orb = orbits.elliptic_orbit(a_list[j], e_list[j])
    
    #Calculate total number of encounters
    Ntotal = min(int(N_cut),int(PB.Ntotal_ecc(Tage, bmax, orb, psi_list[j], b0=0.0))) # This needs to be checked
    Ntotal_list.append(Ntotal)

    
    #print(j, bmax, Ntotal)
    # Added condition to skip if no encounters
    #print(Ntotal)
    if Ntotal == 0:
        M_list_final.append(minicluster.M)
        R_list_final.append(minicluster.R)
        delta_list_final.append(minicluster.delta)
        continue

    if ((Ntotal == N_cut) and (profile == "PL")):
        M_list_final.append(1e-30)
        R_list_final.append(1e-30)
        delta_list_final.append(1e-30)
        continue
    
    
    #Sample properties of the stellar encounters
    blist = PB.dPdb(bmax, Nsamples=Ntotal)
    # print(a_list[j], e_list[j], psi_list[j])
    v_amc_list, r_interaction_list = PB.dPdVamc(orb, psi_list[j], bmax, Nsamples=Ntotal)
    Vlist = PB.dPdV(v_amc_list, PB.sigma(r_interaction_list), Nsamples=Ntotal) 
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
            delta_E = E_pert/minicluster.Ebind()
            #dElist[i] = delta_E
            minicluster.perturb(E_pert)
            
            if (minicluster.M > M_cut):
                #If the density of the AMC increases, recompute the number of encounters required
                if (minicluster.rho_mean() > rho0):
                    #print(dT, T_remain)
                    E_test = PB.Elist(sig_rel, 1.0, Mp, minicluster.M, Rrms2 = minicluster.Rrms2())
                    bmax_new = ((E_test/minicluster.Ebind())*N_cut)**(1./4)
                    
                    
                    N_remain = min(int(N_cut),int(PB.Ntotal_ecc(T_remain, bmax_new, orb, psi_list[j], b0=0.0)))
                    #N_remain = PB.Ntotal_ecc(T_remain, bmax, orb, psi_list[j], b0=0.0)
                    #print(T_remain, bmax, PB.Ntotal_ecc(T_remain, bmax, orb, psi_list[j], b0=0.0), N_remain)
                    
                    if (N_remain == 0):
                        break
                    else:
                        dT = T_remain/N_remain
                    
                    rho0 = minicluster.rho_mean()
                    #blist[(i+1):] = PB.dPdb(bmax, Nsamples=len(blist[(i+1):]))
                    blist[(i+1):] = blist[(i+1):]*(bmax_new/bmax)
                    bmax = bmax_new
            i += 1         
    #print(j, Ntotal, N_enc)         
                    
            
    Nenc_list.append(N_enc)
    M_list_final.append(minicluster.M)
    R_list_final.append(minicluster.R)
    delta_list_final.append(minicluster.delta)
    #
#print("Number of disrupted AMCs:", N_disrupt)
M_list_initial = np.array(M_list_initial)
R_list_initial = np.array(R_list_initial)
delta_list_initial = np.array(delta_list_initial)

M_list_final = np.array(M_list_final)
R_list_final = np.array(R_list_final)
delta_list_final = np.array(delta_list_final)
Ntotal_list = np.array(Ntotal_list)

#------------------------------------ Testing
#print(M_list_initial)
#print(Nenc_list)
#print(M_list_final/M_list_initial)
if (VERBOSE):
    print("p_surv = ", np.sum(M_list_final > 1e-29)/len(M_list_initial))
    print("p_surv (M_f > 10% M_i) = ", np.sum(M_list_final/M_list_initial > 1e-1)/len(M_list_initial))
# print(R_list_initial, R_list_final)

# ecc_str = '_ecc'
# if (circular == True):
#     ecc_str = ''

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,5))
# plt.hist(Ntotal_list, bins=np.linspace(0, 200, num=50))

# plt.xlim(0,100)
# plt.ylim(0,800)
# plt.xlabel("Total Interactions")
# plt.ylabel("$N_\mathrm{AMC}$")
# plt.savefig('../plots/MC_testdistNtot_%.1f%s.pdf'%(Rkpc, ecc_str), bbox_inches='tight')


# plt.figure(figsize=(7,5))
# plt.hist(R_list_initial, alpha = 0.5, bins=np.geomspace(1e-8, 1e-2, 50), label='Initial')
# plt.hist(R_list_final, alpha = 0.5, bins=np.geomspace(1e-8, 1e-2, 50), label='Final')

# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
# plt.xlabel("$R_\mathrm{AMC}$ $[\mathrm{pc}]$")
# plt.ylabel("$N_\mathrm{AMC}$")
# plt.legend(loc='upper right')
# plt.savefig('../plots/MC_testdistR%s.pdf'%(ecc_str,), bbox_inches='tight')

# plt.figure(figsize=(7,5))
# plt.hist(M_list_initial, alpha = 0.5, bins=np.geomspace(M_list_initial.min(), M_list_initial.max(), 50), label='Initial')
# plt.hist(M_list_final, alpha = 0.5, bins=np.geomspace(M_list_initial.min(), M_list_initial.max(), 50), label='Final')

# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
# plt.xlabel("$M_\mathrm{AMC}$ $[M_{\odot}]$")
# plt.ylabel("$N_\mathrm{AMC}$")
# plt.legend(loc='upper right')
# plt.savefig('../plots/MC_testdistM%s.pdf'%(ecc_str,), bbox_inches='tight')
#---------------------------------------------

#print(circular)
if (circular):
    ecc_str = '_circ'
else:
    ecc_str = ''

Results = np.column_stack([M_list_initial, R_list_initial, delta_list_initial, M_list_final, R_list_final, delta_list_final, e_list, psi_list])
if (SAVE_OUTPUT):
    np.savetxt(dirs.montecarlo_dir + 'AMC_logflat_a=%.2f_%s%s.txt'% (a0/1e3, profile, ecc_str), Results, delimiter=', ', header="Columns: M initial [Msun], R initial [pc], Initial overdensity delta,  M final [Msun], R final [pc], Final overdensity delta, eccentricity, psi [rad]")
    np.savetxt(dirs.montecarlo_dir + 'AMC_Ninteractions_a=%.2f_%s%s.txt'% (a0/1e3, profile, ecc_str), Ntotal_list)
    np.savetxt(dirs.montecarlo_dir + 'AMC_Ninteractions_true_a=%.2f_%s%s.txt'% (a0/1e3, profile, ecc_str), Nenc_list)
