#!/usr/bin/env python3

import numpy as np

try:
    from tqdm import tqdm
except ImportError as err:
    def tqdm(x):
        return x

import MC_script_ecc


N_AMC = 100
profile = "test"
IDstr = ""
circular = False

a_list = np.geomspace(0.1, 50, 50) #kpc

for i, a in enumerate(tqdm(a_list, desc="Perturbing miniclusters")):
    MC_script_ecc.Run_AMC_MonteCarlo(a, N_AMC, profile, IDstr, circular)