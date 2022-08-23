import numpy as np

kmtopc = 1.0/(3.086*10**13)
MNS  = 1.4 # Msun
RNS  = 10*kmtopc # pc
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumtrapz
from scipy.special import erfi
from scipy.special import gamma as gamma_fun

import dirs

G    = 4.32275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun



#Average stellar mass
#This number is not really based on anything concrete...
M_star_avg = 1.0 #M_sun

#Bulge distribution from McMillan, P.J. 2011, MNRAS, 414, 2446 1102.4340
# R and Z are spherical coordinate systems in pc
def rho_star_bulge(R, Z):
    r0 = 75.             #pc ### This value has been changed from 750pc to 75pc
    rc = 2.1e3          #pc
    rho_star_core = 99.3 #Msun/pc^3 ### This value has been changed from 200Msun/pc^3 to 100Msun/pc^3
    rp2 = R**2 + 4*Z**2
    rp  = np.sqrt(rp2)

    return rho_star_core*np.exp(-rp2/rc**2)/(1+rp/r0)**(1.8)

def rho_star_disc(R, Z):
    hZt = 0.3e3 #pc
    hRt = 2.9e3 #pc
    
    hZT = 0.9e3 #pc
    hRT = 3.31e3 #pc

    rho_0t = 1.361    #Msun/pc^3 # Changed from  1.57
    rho_0T = 0.116  #Msun/pc^3 # Changed from 0.0546

    Za  = np.abs(Z)
    return rho_0t*np.exp(-R/hRt - Za/hZt) + rho_0T*np.exp(-R/hRT - Za/hZT)

def rho_star_halo(R, Z):
    nH = 2.77
    qH = 0.64
    rho_star_halo = 4.45e-4 #Msun/pc^3 # Changed from 5.25e-5
    
    Rsun = 8.3e3 #pc
    
    return rho_star_halo*(Rsun/np.sqrt(R**2 + (Z/qH)**2))**nH

def rho_star(R, Z):
    return rho_star_bulge(R, Z) + rho_star_disc(R, Z)# + rho_star_halo(R, Z)


#--- Enclose mass and velocity dispersion
## NFW profile for AMC distribution
def rhoNFW(R):
    rho0 =  1.4e7*1e-9 # Msun*pc^-3, see Table 1 in 1304.5127
    rs = 16.1e3      # pc
    aa = R/rs
    return rho0/aa/(1+aa)**2

def M_enc(r):
    rho0 =  1.4e7*1e-9 # Msun pc^-3, see Table 1 in 1304.5127
    rs   = 16.1e3      # pc
    
    #MW mass enclosed within radius a
    Menc = 4*np.pi*rho0*rs**3*(np.log(1+r/rs) - r/(rs+r)) 
    M_BH = 4e6
    return Menc + M_BH

#Velocity dispersion at a given radius r
def sigma(r):
    r_clip = np.clip(r, 1e-20, 1e20)
    return np.sqrt(G*(M_enc(r_clip))/r_clip) # km/s
    
#Local circular speed
def Vcirc(Mstar, rho):
    return np.sqrt(G_pc*(Mstar+M_enc(r))/r) # pc/s
    
#-------------------

OLD = False

if (OLD):

    f_bound = 0.8
    N_bulge = f_bound*6.0e8
    N_disk = f_bound*4.0e8

    #Number densities of neutron stars

    #R_cyl is the cylindrical galactocentric distance
    def nNS_bulge(R_cyl, Z):
        #Bulge distribution from McMillan, P.J. 2011, MNRAS, 414, 2446 1102.4340
        # R_cyl and Z are cylindrical coordinates in pc

        r0 = 75.             #pc ### This value has been changed from 750pc to 75pc
        rc = 2.1e3          #pc
        Nnorm = 1/90218880.  #pc^{-3} - normalising constant so that the distribution integrates to 1 over the whole volume
        rp2 = R_cyl**2 + 4*Z**2
        rp  = np.sqrt(rp2)
        return N_bulge*Nnorm*np.exp(-rp2/rc**2)/(1+rp/r0)**(1.8)

    def nNS_disk(R_cyl, Z):
        #Lorimer profile, Eq. 6 of https://arxiv.org/pdf/1805.11097.pdf
        #Best fit parameters from Table 3 (Broken Power-Law)
        Rsun = 8.5e3 #We use R_sun = 8.5 kpc here for consistency with the fits in 1805.11097
        B = 3.91 
        C = 7.54
        Zs = 0.76e3 
    
        Norm = C**(B+2)/(4*np.pi*Rsun**2*Zs*np.exp(C)*gamma_fun(B+2))
        return N_disk*Norm*(R_cyl/Rsun)**B*np.exp(-C*(R_cyl-Rsun)/Rsun)*np.exp(-np.abs(Z)/Zs)

    def nNS(R_cyl, Z):
        return nNS_bulge(R_cyl, Z) + nNS_disk(R_cyl, Z)


    #Tabulate
    nNS_sph_interp = None

    def calcNS_sph():
        #Galactocentric, spherical R
        r_list = np.geomspace(1, 200e3, 10000)
        nr_list = 0.0*r_list

        for i, r in enumerate(r_list):
            Z_list = 0.99999*np.linspace(-r, r, 1001)
            R_list = np.sqrt(r**2 - Z_list**2)
            nr_list[i] = (0.5/r)*np.trapz(nNS(R_list, Z_list), Z_list)
    
        return interp1d(r_list, nr_list, bounds_error=False, fill_value=0.0)

    #Galactocentric, spherical R
    def nNS_sph(R_sph):
        global nNS_sph_interp
        if (nNS_sph_interp is None):
            nNS_sph_interp = calcNS_sph()
    
        return nNS_sph_interp(R_sph)


    #Parallelised for Z
    def dPdZ(R_sph, Z):
        ma = R_sph**2 > Z**2 #Mask for valid values
        R_cyl = np.sqrt(R_sph**2 - Z[ma]**2)
    
        result = 0.0*Z
        result[ma] = nNS(R_cyl, Z[ma])/(2*R_sph*nNS_sph(R_sph))
        #P(Z) = P(R_sph, Z)/P(R_sph) = (2 pi R_sph n(R_cyl, Z)/(4 pi R_sph^2 <n(R_sph)>))
        return result

else:

    columns = ["B0", "T", "theta","t", "x", "y", "z"]

    data_CMZ = np.load(dirs.NS_data + "Population_Model_CMZ_2.dat")
    CMZ_dict = {columns[i]: data_CMZ[:,i] for i in range(len(columns))}
    CMZ_dict['r'] = np.sqrt(CMZ_dict['x']**2 + CMZ_dict['y']**2 + CMZ_dict['z']**2)

    N_CMZ = len(CMZ_dict['r'])
    P_r_CMZ, r_bins_CMZ = np.histogram(CMZ_dict['r'], bins=100, density=True)
    r_CMZ = 0.5*(r_bins_CMZ[1:] + r_bins_CMZ[:-1])

    data_GC = np.load("../data/Population_Model__FastDecay_Androm_Long.npy")
    GC_dict = {columns[i]: data_GC[:,i] for i in range(len(columns))}
    GC_dict['r'] = np.sqrt(GC_dict['x']**2 + GC_dict['y']**2 + GC_dict['z']**2)

    N_GC = len(GC_dict['r'])
    P_r_GC, r_bins_GC = np.histogram(CMZ_dict['r'], bins=100, density=True)
    r_GC = 0.5*(r_bins_GC[1:] + r_bins_GC[:-1])

    def nNS_CMZ(r):
        return N_CMZ*np.interp(r, r_CMZ, P_r_CMZ/(4*np.pi*r_CMZ**2), left=0.0, right=0.0)

    def nNS_GC(r):
        return N_GC*np.interp(r, r_GC, P_r_GC/(4*np.pi*r_GC**2), left=0.0, right=0.0)
    
    def nNS_sph(r):   # NS distribution at r in pc^-3
        return nNS_CMZ(r) + nNS_GC(r)
    

