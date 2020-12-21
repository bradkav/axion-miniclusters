#!/usr/bin/env python3

import numpy as np
#import matplotlib.pyplot as plt
import AMC
import perturbations as PB
from tqdm import tqdm
import argparse
import sys

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-R", "--radius", type=float, help="Galactocentric Radius [kpc].")
    parser.add_argument("-N", "--AMC_number", type=int, help="Number of AMCs in the simulation.")
    parser.add_argument("-profile", "--profile", type=str, help="Density profile - `PL` or `NFW`", default="PL")
    options = parser.parse_args(args)
    return options

options = getOptions(sys.argv[1:])

R     = options.radius*1e3       # conversion to pc
Rkpc  = options.radius
N_AMC = options.AMC_number
profile = options.profile

# Perturber parameters
Mp = 1

# Here we calculate the number of AMCs in the
# We need to sample the number of AMCs in bins of a given radius
Tage = 4.26e17

sig_rel = PB.sigma(R)

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
M_list, delta_list = PB.sample_AMCs_logflat(m_a = 2e-5, n_samples = N_AMC)
psi_list = np.random.uniform(-np.pi/2.,np.pi/2.,size = N_AMC)


#for j in tqdm(range(N_AMC)):
for j in range(N_AMC):   
    #print(j, M_list[j])
   # Initialise the AMC
    minicluster = AMC.AMC(M = M_list[j], delta = delta_list[j], profile=profile)
    
    M_list_initial.append(minicluster.M)
    R_list_initial.append(minicluster.R)
    delta_list_initial.append(minicluster.delta)
    
    # Galactic parameters
    Vpeak = PB.Vcirc(M_list[j], R)
    n_dist = PB.n(M_list[j], R, psi_list[j])
    E_test = PB.Elist(sig_rel, 1.0, Mp, minicluster.M, Rrms2 = minicluster.Rrms2())
    
    N_cut = 1e7


    #bmax = 1e6*minicluster.R
    bmax = ((E_test/minicluster.Ebind())*N_cut)**(1./4)
    #bmax = 1e6*minicluster.R

    #print("bmax:", bmax)
    #print("1e6*R:", 1e6*minicluster.R)
    
    #print(minicluster.R, bmax)
    #print("Etest/Ebind, bmax", E_test/minicluster.Ebind(),bmax)
    #bmax = minicluster.bmax
    #print("Etest_new/Ebind",  PB.Elist(sig_rel, bmax, Mp, minicluster.M, Rrms2 = minicluster.Rrms2())/minicluster.Ebind())

    Ntotal = min(int(N_cut),int(PB.Ntotal(n_dist, Tage, bmax, Vpeak))) # This needs to be checked
    #print(Ntotal)
    #print(" ")
    #print('nMC = ', PB.nLV(M_list[j], R, psi_list[j]))
    #print('Ntotal = ',Ntotal)
    #print('Vpeak = ',Vpeak)

    blist = PB.dPdb(bmax, Nsamples=Ntotal)
    Vlist = PB.dPdV(sig_rel, Nsamples=Ntotal)
    #print(Vlist)
    # #Choose some cut off mass for the AMC
    M_cut = 1e-20

    Mlist = np.zeros(Ntotal)

    #N_enc = 0
    for i in range(Ntotal):
        #N_enc += 1
        Mlist[i] = minicluster.M
        if (minicluster.M < M_cut):
            #print("    Disrupted after ", N_enc, " encounters")
            #N_disrupt += 1 
            break
        else:
            E_pert = PB.Elist(Vlist[i], blist[i], Mp, minicluster.M, Rrms2 = minicluster.Rrms2()) #Previously Rrms2 = minicluster.R**2/3 for isothermal sphere
            #print(E_pert)
            minicluster.perturb(E_pert)

    M_list_final.append(minicluster.M)
    R_list_final.append(minicluster.R)
    delta_list_final.append(minicluster.delta)
        
#print("Number of disrupted AMCs:", N_disrupt)
M_list_initial = np.array(M_list_initial)
R_list_initial = np.array(R_list_initial)
delta_list_initial = np.array(delta_list_initial)

M_list_final = np.array(M_list_final)
R_list_final = np.array(R_list_final)
delta_list_final = np.array(delta_list_final)


# Results = np.c_[[M_list_initial, R_list_initial, delta_list_initial, M_list_final, R_list_final, delta_list_final]]
np.savetxt('../AMC_data/AMC_logflat_R=%.2f_%s.txt'% (Rkpc, profile), np.column_stack([M_list_initial, R_list_initial, delta_list_initial, M_list_final, R_list_final, delta_list_final]), delimiter=', ', header="# M initial [Msun], R initial [pc], delta initial, M final [Msun], R final [pc], delta final", comments="")
