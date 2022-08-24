import numpy as np
import dirs
import params

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
mass_function_ID = "delta_p"
            
#---------- Calculation Parameters -------
            
N_AMC = 100 #Number of AMCs to simulate for each radius in the Monte Carlos
circular = False
a_list = np.geomspace(1e-2, 50e3, 50) #pc
AScut = False
Ne = 10000 #Number of AMC-NS encounters to generate 

IDstr = "_test"




#---------- Run Monte Carlo Simulations ----------


for i, a in enumerate(tqdm(a_list, desc="Perturbing miniclusters")):
    Run_AMC_MonteCarlo(a*1e-3, N_AMC, m_a, profile, mass_function_ID, galaxyID, circular, IDstr=IDstr)
    
print("> Results saved to " + dirs.montecarlo_dir)


#----------- Prepare distributions ---------------

prepare_distributions(m_a, profile, mass_function_ID, galaxyID, circular, IDstr=IDstr)


#----------- Plot survival probabilities ----------

PlotSurv.plot_psurv_a(profile, mass_function_ID, IDstr, save_plot=True, show_plot=False)
PlotSurv.plot_psurv_r(profile, mass_function_ID, IDstr, circular=False, save_plot=True, show_plot=False)
PlotSurv.plot_encounter_rate(profile, mass_function_ID,  IDstr, circular=False, save_plot=True, show_plot=True)

#-----------Sample encounters ----------------------

sample_encounters(Ne, m_a, profile,  mass_function_ID, galaxyID, circular=circular, AScut = AScut, IDstr=IDstr)