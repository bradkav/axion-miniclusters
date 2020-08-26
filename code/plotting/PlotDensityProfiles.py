import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

rc('text', usetex=True)
rc('font', size=20)


c = 100
def f_NFW(x):
    return np.log(1+x) - x/(1+x)

def R_NFW(M_AMC, rho_AMC):
    r_s = (M_AMC/(4*np.pi*rho_AMC*f_NFW(c)))**(1/3)
    return c*r_s

def rho_NFW(r, M_AMC, rho_AMC):
    r_s = (M_AMC/(4*np.pi*rho_AMC*f_NFW(c)))**(1/3)
    x = (r/r_s)
    return rho_AMC/(x*(1+x)**2)

def R_PL(M_AMC, rho_AMC):
    return (3*M_AMC/(4*np.pi*rho_AMC))**(1/3)

def rho_PL(r, M_AMC, rho_AMC):
    R_AMC = R_PL(M_AMC, rho_AMC)
    return 0.25*rho_AMC*(R_AMC/r)**(9/4)
    


#-------------------

plt.figure(figsize=(7,5))

R_list = np.geomspace(1e-7, 1e0, 1000)

M0 = 1e-10
rho0 = 10

Rmax_PL = R_PL(M0, rho0)
Rmax_NFW = R_NFW(M0, rho0) 

#plt.axhline(1, linestyle='--', color='k')

#plt.loglog(de, dm, label='Power-law')
plt.loglog(R_list[R_list < Rmax_PL], rho_PL(R_list[R_list < Rmax_PL], M0, rho0), label = 'Power-law', color='C0')
plt.loglog([Rmax_PL, Rmax_PL], [1e-10 , rho_PL(Rmax_PL, M0, rho0)], linestyle='--', color='C0')

plt.loglog(R_list[R_list < Rmax_NFW], rho_NFW(R_list[R_list < Rmax_NFW], M0, rho0), label = 'NFW', color='C8')
plt.loglog([Rmax_NFW, Rmax_NFW], [1e-10 , rho_NFW(Rmax_NFW, M0, rho0)], linestyle='--', color='C8')

plt.xlabel(r"$R$ [pc]")
plt.ylabel(r"$\rho_\mathrm{int}(R) \,[M_\odot \,\mathrm{pc}^{-3}]$")

plt.legend(loc='upper right')

plt.ylim(1e-7, 1e8)
plt.xlim(1e-7, 1e-2)

plt.text(0.68, 0.75, "$M_\\mathrm{AMC} = 10^{-10}\\,M_\\odot$\n$\\rho_\\mathrm{AMC} = 10 \\,M_\\odot\\,\\mathrm{pc}^{-3}$", transform=plt.gca().transAxes, ha='left', va = 'top', fontsize=14)

plt.gca().set_xticks(np.geomspace(1e-7, 1e-2, 6))
#plt.gca().set_xticklabels([], minor=True)

plt.savefig("../../plots/AMC_densityprofiles.pdf", bbox_inches='tight')

plt.show()