#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import interp1d

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
    
def inverse_transform_sampling(function, x_range, nbins=1000, n_samples=1000, logarithmic=False):
    if (logarithmic):
        bins = np.geomspace(x_range[0], x_range[-1], num=nbins)
    else:
        bins = np.linspace(x_range[0], x_range[-1], num=nbins)
    pdf = function(np.delete(bins,-1) + np.diff(bins)/2)
    Norm = np.sum(pdf*np.diff(bins))
    pdf /= Norm
    cumul_values = np.zeros(bins.shape)
    cumul_values[1:] = np.cumsum(pdf*np.diff(bins))
    inv_cdf = interp1d(cumul_values, bins)
    r = np.random.rand(n_samples)
    return inv_cdf(r)