#!/usr/bin/env python3

import numpy as np
import params


try:
    from tqdm import tqdm
except ImportError as err:
    def tqdm(x):
        return x

import MC_script_ecc


N_AMC = 100
profile = "NFW"
IDstr = params.IDstr
circular = False

a_list = np.geomspace(1e-2, 50e3, 50) #pc

for i, a in enumerate(tqdm(a_list, desc="Perturbing miniclusters")):
    MC_script_ecc.Run_AMC_MonteCarlo(a*1e-3, N_AMC, profile, IDstr, circular)