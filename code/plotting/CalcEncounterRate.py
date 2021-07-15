import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

rc('text', usetex=True)
rc('font', size=18)

root_dir = '../../data/'

frac_AScut_NFW = 0.0146239
frac_AScut_PL = 0.0002719

#IDstr = "_gamma-0.5"
IDstr = "_ma_480mueV"
#IDstr = "_wStripping"

#To-Do - make the correction for the initial range of samples masses more concrete

if (IDstr ==  "_gamma-0.5"):
    frac_AScut_NFW = 0.0463458
    frac_AScut_PL = 0.002647
elif (IDstr == "_wStripping"):
    #frac_AScut_NFW = 0.0138*0.9332610321836602
    frac_AScut_NFW = 0.0146239*0.8764842147261401
    frac_AScut_PL = 0.0002719*0.8764842147261401
else:
    frac_AScut_NFW = 0.0146239
    frac_AScut_PL = 0.0002719



AS_CUT = True

cut_text = ""
if (AS_CUT):
    print("> Calculating with axion-star cut...")
    cut_text = "_AScut"
    print("> Initial fraction of AMCs surviving AS cut:")
    print(">    NFW:", frac_AScut_NFW)
    print(">    PL:", frac_AScut_PL)



#------ Plot encounter rate as a function of R ----------------------------

print("Calculating for: %s..."%(IDstr,))

Rlist_PL, PDF_R_PL = np.loadtxt(root_dir + 'EncounterRate_PL%s%s.txt'%(cut_text,IDstr), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)
Rlist_NFW, PDF_R_NFW = np.loadtxt(root_dir + 'EncounterRate_NFW%s%s.txt'%(cut_text,IDstr), delimiter =',', dtype='f8', usecols=(0,1), unpack=True)

Gamma_PL = np.trapz(PDF_R_PL, Rlist_PL)*3600*24
Gamma_NFW = np.trapz(PDF_R_NFW, Rlist_NFW)*3600*24

print("    Gamma_PL [/day]:", Gamma_PL)
print("    Gamma_NFW [/day]:", Gamma_NFW)
