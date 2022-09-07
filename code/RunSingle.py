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

#200*5000*14 -> 30 minutes
            
#---------- Calculation Parameters -------
            
N_AMC = 1000 #Number of AMCs to simulate for each radius in the Monte Carlos
circular = False
a_list = np.geomspace(1e-2, 50e3, 50) #pc
AScut = False
Ne = 10000 #Number of AMC-NS encounters to generate 

IDstr = "_M31_youngNS_delta_10"

#print(mass_function.rho_of_delta(1.0))

#You can use the function "get_mass_function" to build
#a mass function based on a few different possible
#"mass_function_ID" values.
#mass_function_ID = "delta_p"
#mass_function_ID = "powerlaw"
#AMC_MF = mass_function.get_mass_function(mass_function_ID, m_a, profile)

#If you add a label to the mass function object, it will be added to the
#names of the output files
#AMC_MF.label = mass_function_ID

#Alternatively, you could just build a mass function from scratch
#with whatever properties you want.
#AMC_MF = mass_function.DeltaMassFunction(m_a=m_a, M0=1e-10)
#AMC_MF.label = "MyMassFunction"

#---------- Run Monte Carlo Simulations ----------

def run_AMC_mass(M_AMC):
    AMC_MF = mass_function.DeltaMassFunction(m_a=m_a, M0=M_AMC, delta_min=9.9, delta_max=10.1)
    AMC_MF.label = f"M_AMC_{M_AMC:.2e}"

    
    print(AMC_MF.label)

    for i, a in enumerate(tqdm(a_list, desc="> Perturbing miniclusters")):
        Run_AMC_MonteCarlo(a*1e-3, N_AMC, m_a, profile, AMC_MF, galaxyID, circular, IDstr=IDstr)
    
    print("> Results saved to " + dirs.montecarlo_dir)


    #----------- Prepare distributions ---------------

    Gamma, Gamma_AScut = prepare_distributions(m_a, profile, AMC_MF, galaxyID, circular, IDstr=IDstr)


    #----------- Plot survival probabilities ----------

    #PlotSurv.plot_psurv_a(profile, AMC_MF, IDstr, save_plot=True, show_plot=False)
    #PlotSurv.plot_psurv_r(profile, AMC_MF, IDstr, circular=False, save_plot=True, show_plot=False)
    #PlotSurv.plot_encounter_rate(profile, AMC_MF,  IDstr, circular=False, save_plot=True, show_plot=True)

    #-----------Sample encounters ----------------------

    T_enc_list = sample_encounters(Ne, m_a, profile,  AMC_MF, galaxyID, circular=circular, AScut = AScut, IDstr=IDstr)
    
    T_lower, T_med, T_upper = np.percentile(T_enc_list, [15.9, 50, 84.1])
    
    return Gamma, Gamma_AScut, T_lower, T_med, T_upper
    
# Need to fix code to load from file it simulation is already done
M_list = np.array([1e-18])
#print(M_list)

gamma_list = 0.0*M_list
gamma_AS_list = 0.0*M_list
T_lower_list = 0.0*M_list
T_med_list = 0.0*M_list
T_upper_list = 0.0*M_list

for i, M in enumerate(M_list):
    gamma_list[i], gamma_AS_list[i], T_lower_list[i], T_med_list[i], T_upper_list[i] = run_AMC_mass(M)
    
np.savetxt("../data/MassGrid" + IDstr + ".txt", np.c_[M_list, gamma_list, gamma_AS_list, T_lower_list, T_med_list, T_upper_list])
print("> Done")
    

