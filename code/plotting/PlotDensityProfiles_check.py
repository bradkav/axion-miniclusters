import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from scipy.interpolate import interp1d

rc('text', usetex=True)
rc('font', size=20)


G_N = 4.301e-3 #(km/s)^2 pc/Msun

z_eq = 3400
rho_eq = 1512.0 #Solar masses per pc^3

#c = 100
def f_NFW(x):
    return np.log(1+x) - x/(1+x)

def R_NFW(M_AMC, rho_AMC, c=100):
    r_s = (M_AMC/(4*np.pi*rho_AMC*f_NFW(c)))**(1/3)
    return c*r_s

def rho_NFW(r, M_AMC, rho_AMC, c=100):
    r_s = (M_AMC/(4*np.pi*rho_AMC*f_NFW(c)))**(1/3)
    x = (r/r_s)
    return rho_AMC/(x*(1+x)**2)

def R_PL(M_AMC, rho_AMC):
    return (3*M_AMC/(4*np.pi*rho_AMC))**(1/3)

def rho_PL(r, M_AMC, rho_AMC):
    R_AMC = R_PL(M_AMC, rho_AMC)
    return 0.25*rho_AMC*(R_AMC/r)**(9/4)
    
rho_crit = 1.29e-7
    
c_list = np.geomspace(1, 100000, 100000)
x_out_list = f_NFW(c_list)/c_list**3
interpx = interp1d(x_out_list, c_list)

#Checking with a different definition of rho_AMC
def rho_NFW_v2(r, M_AMC, rho_AMC):
    rho_s = rho_AMC/0.58
    #rho_s = rho_AMC
    #R_vir = (3*(rho_crit*200)/(4*np.pi*M_AMC))**(1/3)
    R_vir = (3*M_AMC/(4*np.pi*rho_crit*200))**(1/3)
    c_v2 = interpx((rho_crit*200)/(3*rho_s))
    print(c_v2)
    r_s = R_vir/c_v2
    #r_s = (M_AMC/(4*np.pi*rho_s*f_NFW(c)))**(1/3)
    x = (r/r_s)
    return rho_s/(x*(1+x)**2) 
    
def R_NFW_v2(M_AMC, rho_AMC):
    rho_s = rho_AMC/0.58
    R_vir = (3*M_AMC/(4*np.pi*rho_crit*200))**(1/3)
    return R_vir



#-------------------

plt.figure(figsize=(7,5))

R_list = np.geomspace(1e-8, 1e3, 1000)

M0 = 1e-10
rho0 = 1e6

Ma = M0/2.26496

Rmax_PL = R_PL(M0, rho0)
Rmax_NFW = R_NFW(Ma, rho0) 
print("Mean density NFW:", 3*M0/(4*np.pi*Rmax_NFW**3))
Rmax_NFW_v2 = R_NFW_v2(M0, rho0) 
print("Mean density NFWd:", 3*M0/(4*np.pi*Rmax_NFW_v2**3))
Rmax_NFWa = R_NFW(M0, rho0, c = 10000)

#plt.axhline(1, linestyle='--', color='k')

#plt.loglog(de, dm, label='Power-law')
plt.loglog(R_list[R_list < Rmax_PL], rho_PL(R_list[R_list < Rmax_PL], M0, rho0), label = 'Power-law', color='C0')
plt.loglog([Rmax_PL, Rmax_PL], [1e-10 , rho_PL(Rmax_PL, M0, rho0)], linestyle='--', color='C0')

plt.loglog(R_list[R_list < Rmax_NFW], rho_NFW(R_list[R_list < Rmax_NFW], Ma, rho0), label = r'NFW ($c=100$)', color='C8')
plt.loglog([Rmax_NFW, Rmax_NFW], [1e-10 , rho_NFW(Rmax_NFW, Ma, rho0)], linestyle='--', color='C8')

plt.loglog(R_list[R_list < Rmax_NFWa], rho_NFW(R_list[R_list < Rmax_NFWa], M0, rho0, c=10000), label = r'NFW ($c=10000$)', color='C9')
plt.loglog([Rmax_NFWa, Rmax_NFWa], [1e-10 , rho_NFW(Rmax_NFWa, M0, rho0, c=10000)], linestyle='--', color='C9')

plt.xlabel(r"$R$ [pc]")
plt.ylabel(r"$\rho_\mathrm{int}(R) \,[M_\odot \,\mathrm{pc}^{-3}]$")

plt.legend(loc='upper right')

plt.ylim(1e-8, 1e9)
plt.xlim(1e-7, 1e-1)

plt.text(0.6, 0.65, "$M_\\mathrm{AMC} = 10^{-10}\\,M_\\odot$\n$\\rho_\\mathrm{AMC} = 10^6 \\,M_\\odot\\,\\mathrm{pc}^{-3}$", transform=plt.gca().transAxes, ha='left', va = 'top', fontsize=14)

#plt.gca().set_xticks(np.geomspace(1e-7, 1e-2, 6))
#plt.gca().set_xticklabels([], minor=True)

plt.savefig("../../plots/AMC_densityprofiles_NFWcompare.pdf", bbox_inches='tight')

plt.show()
