import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import sys

sys.path.append('../')

import AMC

rc('text', usetex=True)
rc('font', size=20)

#-------------------

plt.figure(figsize=(7,5))

plt.axhline(1, linestyle='--', color='k')

plt.loglog([1e-12, 1e-12], [1e-12, 1e-12], 'k-', label='$\Delta M/M$')
plt.loglog([1e-12, 1e-12], [1e-12, 1e-12], 'k:', label='$f_\mathrm{ej}$')

dE_PL, dM_PL = np.loadtxt("../../data/MassLoss_PL.txt", unpack=True)
dE_NFW, dM_NFW = np.loadtxt("../../data/MassLoss_NFW.txt", unpack=True)

plt.loglog(dE_PL, dM_PL, linestyle ='-', label = 'Power-law', color='C0')
plt.loglog(dE_NFW, dM_NFW, linestyle = '-', label = 'NFW', color='C8')

dE_PL, fej_PL = np.loadtxt("../../data/EnergyLoss_PL.txt", unpack=True)
dE_NFW, fej_NFW = np.loadtxt("../../data/EnergyLoss_NFW.txt", unpack=True)

plt.loglog(dE_PL, fej_PL, linestyle =':', color='C0')
plt.loglog(dE_NFW, fej_NFW, linestyle = ':', color='C8')

plt.xlabel(r"$\Delta E/E_\mathrm{bind}$")
#plt.ylabel(r"$\Delta M/M$")

plt.legend(loc=[0.02, 0.55], fontsize=16)

plt.ylim(1e-6, 2)
plt.xlim(1e-5, 1e3)

plt.gca().set_xticks(np.geomspace(1e-5, 1e3, 9))
#plt.gca().set_xticklabels([], minor=True)
plt.gca().set_yticks(np.geomspace(1e-6, 1e0, 7), minor=True)
plt.gca().set_yticklabels([], minor=True)

plt.savefig("../../plots/MassLoss.pdf", bbox_inches='tight')


#--------------------------


for profile in ["PL", "NFW"]:

    mc = AMC.AMC(M=1e-10, delta=1.0, profile=profile)

    Npert = 1000

    Mlist = np.zeros(Npert)
    Rlist = np.zeros(Npert)
    Elist = np.zeros(Npert)

    #Perturbation which is 1 permille of the binding energy
    dE = 1e-3*mc.Ebind()
    #print(dE)
    for i in range(Npert):
        Mlist[i] = mc.M
        Rlist[i] = mc.R
        Elist[i] = mc.Ebind()
    
        if (mc.M > 1e-25):
            #Inject an energy dE into the minicluster
            mc.perturb(dE)


    #print("here!")
    plt.figure(figsize=(7,5))

    plt.axhline(1., linestyle='--', color='grey')

    plt.semilogy(Mlist/Mlist[0], label=r'$M/M_i$', color='C1')
    plt.plot(Rlist/Rlist[0], label=r'$R/R_i$', color='C2')
    plt.plot(Elist/Elist[0], label=r'$E_\mathrm{bind}/E_{\mathrm{bind},i}$', color='C3')

    plt.ylabel(r"$x/x_i$")
    plt.xlabel(r"Number of perturbations")

    plt.legend()
    plt.ylim(1e-3, 1e3)
    plt.xlim(0, 600)

    if (profile == "NFW"):
        profile_text = "NFW"
    elif (profile == "PL"):
        profile_text = "Power-law"

    plt.title(profile_text + " profile, $\Delta E/E_{\mathrm{bind},i} = 10^{-3}$")

    plt.savefig("../../plots/Perturbations_" + profile + ".pdf", bbox_inches="tight")



plt.show()