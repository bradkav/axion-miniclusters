import numpy as np
import dirs
import params

import mass_function

from MonteCarlo import Run_AMC_MonteCarlo
from prepare_distributions import prepare_distributions
from plotting import PlotSurvivalProbability as PlotSurv
from simulate_signal import sample_encounters

try:
    from tqdm import tqdm
except ImportError as err:
    def tqdm(x):
        return x
        
#---------- Model Parameters ------------
 
m_a = 35.16e-6           
#m_a = 16.54e-6
#m_a = 1e-6
profile = "PL"
galaxyID = "M31"

            
#---------- Calculation Parameters -------
            
N_AMC = 10 #Number of AMCs to simulate for each radius in the Monte Carlos
circular = False
a_list = np.geomspace(1e-2, 50e3, 50) #pc
AScut = False
Ne = 1000 #Number of AMC-NS encounters to generate 

IDstr = "_test"

#You can use the function "get_mass_function" to build
#a mass function based on a few different possible
#"mass_function_ID" values.
mass_function_ID = "delta_p"
#mass_function_ID = "powerlaw"
AMC_MF = mass_function.get_mass_function(mass_function_ID, m_a, profile)

#If you add a label to the mass function object, it will be added to the
#names of the output files
AMC_MF.label = mass_function_ID

#Alternatively, you could just build a mass function from scratch
#with whatever properties you want.
#AMC_MF = mass_function.DeltaMassFunction(m_a=m_a, M0=1e-10)
#AMC_MF.label = "MyMassFunction"

#---------- Run Monte Carlo Simulations ----------


for i, a in enumerate(tqdm(a_list, desc="> Perturbing miniclusters")):
    Run_AMC_MonteCarlo(a*1e-3, N_AMC, m_a, profile, AMC_MF, galaxyID, circular, IDstr=IDstr)
    
print("> Results saved to " + dirs.montecarlo_dir)


#----------- Prepare distributions ---------------

prepare_distributions(m_a, profile, AMC_MF, galaxyID, circular, IDstr=IDstr)


#----------- Plot survival probabilities ----------

PlotSurv.plot_psurv_a(profile, AMC_MF, IDstr, save_plot=True, show_plot=False)
PlotSurv.plot_psurv_r(profile, AMC_MF, IDstr, circular=False, save_plot=True, show_plot=False)
PlotSurv.plot_encounter_rate(profile, AMC_MF,  IDstr, circular=False, save_plot=True, show_plot=True)

#-----------Sample encounters ----------------------

sample_encounters(Ne, m_a, profile,  AMC_MF, galaxyID, circular=circular, AScut = AScut, IDstr=IDstr)