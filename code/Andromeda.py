import numpy as np

import dirs

G    = 4.32275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun

#Average stellar mass
#This number is not really based on anything concrete...
M_star_avg = 1.0 #M_sun

# Bulge distribution from Tamm et al. 1208.5712
# R and Z are spherical coordinate systems in pc
def rho_star_bulge(R, Z):
    rc = 2.025e3         #pc
    dN = 11.67
    q  = 0.73
    rho_star_core = 0.22 #Msun/pc^3
    rp  = np.sqrt(R**2 + (Z/q)**2)
    # Einasto profile, Eq.2 of 1208.5712
    return rho_star_core*np.exp(-dN*((rp/rc)**0.25 - 1.))

def rho_star_disc(R, Z):
    rc = 11.35e3         #pc
    dN = 2.67
    q  = 0.1
    rho_star_disc = 0.0172 #Msun/pc^3
    rp  = np.sqrt(R**2 + (Z/q)**2)
    return rho_star_disc*np.exp(-dN*(rp/rc - 1.))

def rho_star(R, Z):
    return rho_star_bulge(R, Z) + rho_star_disc(R, Z)



#----------- Enclosed mass and velocity dispersion
def rhoNFW(R):
    rho0 = 5.0e6 * 1e-9  # Msun*pc^-3, see astro-ph/0110390
    rs = 25.0e3  # pc, see astro-ph/0110390 table 3 using virial radius/concentration
    aa = R / rs
    return rho0 / aa / (1 + aa) ** 2

def M_enc(r):
    rho0 = 5.0e6 * 1e-9  # Msun pc^-3, see astro-ph/0110390
    rs = 25.0e3  # pc
    M = 4 * np.pi * rho0 * rs ** 3 * (np.log((rs + r) / rs) - r / (rs + r))
    M_BH = 3e7
    return M + M_BH

# Velocity dispersion at a given radius rho
def sigma(r):
    r_clip = np.clip(r, 1e-20, 1e20)
    return np.sqrt(G * (M_enc(r_clip)) / r_clip)  # km/s

# Local circular speed
def Vcirc(Mstar, r):
    return np.sqrt(G_pc * (Mstar + Menc(r)) / r)  # pc/s


#-----------------


#-------------


columns = ["B0", "T", "theta","t", "x", "y", "z"]


    
#print("Reading XXX_3.dat...")
NS_fname = "Population_Model_CMZ_2.dat"
#Use the following line for a very young NS sample:
#NS_fname = "Population_Model_CMZ_3.dat"

#print("Reading " + NS_fname)
data_CMZ = np.load(dirs.NS_data + NS_fname)
CMZ_dict = {columns[i]: data_CMZ[:,i] for i in range(len(columns))}

print("> Adjusting NS distributions for Andromeda!")
for key in ["x", "y", "z"]:
    CMZ_dict[key] *= 25.0/16.1
CMZ_dict['r'] = np.sqrt(CMZ_dict['x']**2 + CMZ_dict['y']**2 + CMZ_dict['z']**2)
len_CMZ = len(CMZ_dict['r'])

N_CMZ = len_CMZ
N_CMZ *= 1/0.6
P_r_CMZ, r_bins_CMZ = np.histogram(CMZ_dict['r'], bins=100, density=True)
r_CMZ = 0.5*(r_bins_CMZ[1:] + r_bins_CMZ[:-1])

P_z_CMZ, z_bins_CMZ = np.histogram(CMZ_dict['z'], bins=100, density=True)
z_CMZ = 0.5*(z_bins_CMZ[1:] + z_bins_CMZ[:-1])


#print("Number of CMZ NSs:", N_CMZ)

data_GC = np.load("../data/Population_Model__FastDecay_Androm_Long.npy")
GC_dict = {columns[i]: data_GC[:,i] for i in range(len(columns))}
for key in ["x", "y", "z"]:
    GC_dict[key] *= 25.0/16.1
GC_dict['r'] = np.sqrt(GC_dict['x']**2 + GC_dict['y']**2 + GC_dict['z']**2)
len_GC = len(GC_dict['r'])

N_GC = len_GC
N_GC *= 1/0.6
P_r_GC, r_bins_GC = np.histogram(CMZ_dict['r'], bins=100, density=True)
r_GC = 0.5*(r_bins_GC[1:] + r_bins_GC[:-1])

P_z_GC, z_bins_GC = np.histogram(GC_dict['z'], bins=100, density=True)
z_GC = 0.5*(z_bins_GC[1:] + z_bins_GC[:-1])

N_tot = N_GC + N_CMZ

def nNS_CMZ(r):
    return N_CMZ*np.interp(r, r_CMZ, P_r_CMZ/(4*np.pi*r_CMZ**2), left=0.0, right=0.0)

def nNS_GC(r):
    return N_GC*np.interp(r, r_GC, P_r_GC/(4*np.pi*r_GC**2), left=0.0, right=0.0)
    
def nNS_sph(r):   # NS distribution at r in pc^-3
    return nNS_CMZ(r) + nNS_GC(r)
    
def dPdZ(z):
    return (N_GC/N_tot)*np.interp(z, z_GC, P_z_GC, left=0.0, right=0.0) + (N_CMZ/N_tot)*np.interp(z, z_CMZ, P_z_CMZ, left=0.0, right=0.0)
