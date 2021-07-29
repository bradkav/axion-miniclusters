import numpy as np

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
