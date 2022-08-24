#!/usr/bin/env python3
import numpy as np

#Generate an ID suffix which can label the filenames
def generate_suffix(profile, mass_function_ID, circular=False, AScut=False, unperturbed=False,IDstr="", verbose=False):
    
    #Prepare file ID strings
    circ_text = ""
    if circular:
        if verbose: print("> Calculating for circular orbits...")
        circ_text = "_circ"

    cut_text = ""
    if AScut:
        if (verbose): print("> Calculating with axion-star cut...")
        cut_text = "_AScut"
        
    pert_text = ""
    if (unperturbed):
        if verbose: print("> Calculating unperturbed distributions...")
        pert_text = "_unpert"
    
 
    return profile + "_" + mass_function_ID + circ_text + pert_text + IDstr
    
#Axion star radius
def r_AS(M_AMC, m_a):
    m_22 = m_a / 1e-22
    return 1e3 * (1.6 / m_22) * (M_AMC / 1e9) ** (-1 / 3)