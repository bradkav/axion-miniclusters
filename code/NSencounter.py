import numpy as np
import perturbations  as PB
import math
G    = 4.32275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun
kmtopc = 1.0/(3.086*10**13)
MNS  = 1.0 # Msun
RNS  = 1.4*kmtopc # pc
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumtrapz
from scipy.special import erfi

##
##  NS distributions
##


def f_NFW(x):
    return np.log(1+x) - x/(1+x)

def density(mass, radius):
        return 3.*mass/(4.*np.pi*radius**3)

def del_density(mass, radius):
    return 3.*density(mass, radius)/radius

def MCradius(mass, density):
    return (3.*mass/(4.*np.pi*density))**(1./3.)

# Interpolation function for radius as a function of NFW density
#--------------------------------------------------------------
c = 100
x_list = np.geomspace(1e-5, 1e5, 1000) # x = r/R_max = r/(c r_s)
rho_list = 1/((c*x_list)*(1+c*x_list)**2)
x_of_rho_interp = interp1d(rho_list, x_list, bounds_error=False, fill_value = 0.0)
def x_of_rho(rho):
    x = x_of_rho_interp(rho)
    m1 = rho > np.max(rho_list)
    x[m1] = c**-1/(rho[m1])
    m2 = rho < np.min(rho_list)
    x[m2] = c**-1/(rho[m2])**(1/3)
    return x    
#--------------------------------------------------------------

#R_cyl is the cylindrical galactocentric distance
def nNS(R_cyl, Z):
    # Distribution of NS in the bulge in pc^-3 from Hernquist Astrophys.J.356 359 (1990)
    Nb = 6.0e8
    ab = 0.6e3       #pc
    rp2 = R_cyl**2 + Z**2
    rp  = np.sqrt(rp2)
    n_bulge = Nb/(2.0*np.pi)*(ab/rp)/(ab+rp)**3

    # Distribution of NS in the disc in pc^-3 from 0904.3102
    Nd = 4.0e8
    sz = 1.0e3  #pc
    ss = 5.0e3  #pc
    nd = Nd/(4.0*np.pi*sz*ss**2)
    Zp = np.abs(Z)
    n_disc = nd*np.exp( -0.5*(R_cyl/ss)**2 - Zp/sz )

    return n_bulge + n_disc

# Integrate over \phi in spherical coordinates
def nNS_sph(R):
    Nb = 6.0e8
    ab = 0.6e3  #pc
    n_bulge = Nb/(2.0*np.pi)*(ab/R)/(ab+R)**3

    Nd = 4.0e8
    sz = 1.0e3  #pc
    ss = 5.0e3  #pc
    n0 = Nd/(4.0*np.sqrt(2*np.pi)*sz*ss*R)
    n_disc = n0*np.exp(-0.5*(R/ss)**2-0.5*(ss/sz)**2)
    n_disc *= (erfi(ss/(np.sqrt(2)*sz)) + erfi((R*sz - ss**2)/(np.sqrt(2)*ss*sz)))
    #n_disc = n0*2.0*np.exp(-0.5*(R/ss)**2)*np.sqrt(np.pi)*ss*sz
    #n_disc = n_disc*math.erf(R*np.sqrt(ss**2 - sz**2/2.0)/ss/sz)
    #n_disc = n_disc/(R*np.sqrt((2.0*ss)**2 - 2.0*sz**2))

    return n_bulge + n_disc

#R here is the 'spherical' galactocentric distance
def n_NS_total(R, Z):
    Nb = 6.0e8
    ab = 0.6e3
    n_bulge = Nb/(2.0*np.pi)*(ab/R)/(ab+R)**3
    Nd = 4.0e8
    sz = 1.0e3
    ss = 5.0e3
    nd = Nd/(4.0*np.pi*sz*ss**2)
    return n_bulge + nd*np.exp( -0.5*(R**2-Z**2)/ss**2 - (np.abs(Z)/sz))

def dPdZ(R, Z):
    # FIXME: This needs to be corrected to the new function
    return n_NS_total(R, Z)/(2*R*nNS_sph(R))

#    Zaxis = np.linspace(-r, r, num=101, endpoint=True)
#    PDF = lambda Z: n_bulge + nd*np.exp( -0.5*(r**2-Z**2)/ss**2 - (Z/sz)**2 ) 
#    return PDF(Zp) / np.trapz(PDF(Zaxis), Zaxis)

## NFW profile for AMC distribution
def rhoNFW(R):
    rho0 =  1.4e7*1e-9 # Msun*pc^-3, see Table 1 in 1304.5127
    rs = 16.1e3      # pc
    aa = R/rs
    return rho0/aa/(1+aa)**2

## AMC distributions

#HMF which is vectorised...
def HMF(mass, mmin, mmax, gg):
    # Halo Mass Function 
    g1   = 1.-gg
    ff = g1*mass**g1/(mmax**g1-mmin**g1)
    ff[mass > mmax] = 0.
    ff[mass < mmin] = 0.
    return ff

# FIXME: What is the different between there. Remove...

@np.vectorize
def HMF_sc(mass, mmin, mmax, gg):
    # Halo Mass Function 
    g1   = 1.-gg
    ff = 1e-40 # FIXME: Check whether changing this number affects anything
    if mass < mmax and mass > mmin:
        ff = g1*mass**g1/(mmax**g1-mmin**g1)
    return ff


def P_r_given_rho(R, rho, mmin, mmax, gg):
    mass = 4.*np.pi*rho*R**3/3.
    # print('made it here', HMF_sc(mass, mmin, mmax, gg), mass, mmin, mmax, gg)
    # quit()
    return 3.*mass/R*HMF_sc(mass, mmin, mmax, gg)/mass

##
## Cross-section
##

def sigma_grav(R_AMC):
    # The velocity dispersion is Maxwell-Boltzmann with dispersion sigma_u=290km/s
    # The cross-section is \pi(R^2)(1+sigma_G2/u^2) with sigma_G2 = 2GM/R
    # The velocity-averaged cross section is \sqrt(2/pi/sigma_u^2)(sigma_G +2sigma_u^2)
    sigma_G2 = 2.0*G*MNS/(RNS + R_AMC) # (km/s)**2
    sigma_u2 = 290.0**2  # (km/s)**2
    ## Returns the cross section*u in pc^3/s
    return (RNS**2 + R_AMC**2)*np.sqrt(2.0*np.pi/sigma_u2)*(sigma_G2 +2.0*sigma_u2)*kmtopc

def Vcirc(rho):
    rho0 =  1.4e7*1e-9 # Msun pc^-3, see Table 1 in 1304.5127
    rs   = 16.1e3      # pc
    Menc = 4*np.pi*rho0*rs**3*(np.log((rs+rho)/rs) - rho/(rs+rho))
    return np.sqrt(G_pc*Menc/rho) # pc/s

def MC_profile(delta, r):
    # r is in units of the axion MC
    # Return the density of axion MC in GeV/pc^3
    hbarc = 0.197e-13 # GeV*cm
    pc    = 3.086e18  # cm to pc
    rhoeq = 2.45036e-37*(pc/hbarc)**3  # GeV/pc^3
    rhoc  = 140.*(1. + delta)*delta**3*rhoeq
    return 0.25*rhoc/r**(9/4)

#def MC_profile_self(M, R, r):
#    ## r is in units of the axion MC radius
#    ## M is in units of M_Sun
#    ## R is in pc
#    ## Returns the density of axion MCs in GeV/pc^3
#    MSuninGeV = 1.115e57
#    rho0 = 1.3187e62 # Density of axion MC in GeV/pc^3
#    return 3.*M/(4.*np.pi*R**3)*MSuninGeV/r**(9/4)

def MC_profile_self(density, r):
    ## r is in units of the axion MC radius
    ## Returns the density of axion MCs in GeV/pc^3
    MSuninGeV = 1.115e57
    #Factor of 0.25 because density at surface is rho/4
    return 0.25*density*MSuninGeV/r**(9/4)
    
def MC_profile_NFW(density, r):
    c = 100
    MSuninGeV = 1.115e57
    rho_s = density*c**3/(3*f_NFW(c))
    return rho_s*MSuninGeV/(c*r*(1+c*r)**2)
    
    
def rc(theta, B0, P0, mhz):
    ## code Eq.5 in 1804.03145
    ## Returns the conversion radius in pc
    rc0 = 7.25859e-12 # pc
    ft  = np.abs(3.*np.cos(theta)**2 - 1)
    return rc0*(ft*B0/1.e14*1/P0/mhz**2)**(1./3.)

def signal(theta, Bfld, Prd, density, fa, ut, s0, r, ret_bandwidth=False, profile = "PL"):
    # Returns the expected signal in microjansky
    cs          = 3.0e8     # speed of light in m/s
    pc          = 3.0860e16 # pc in m
    hbar        = 6.582e-16 # GeV/GHz
    hbarT       = 6.582e-25 # GeV s
    GaussToGeV2 = 1.953e-20 # GeV^2
    alpEM       = 1/137.036 # Fine-structure constant
    Lambda      = 0.0755    # confinment scale in GeV

    ma     = Lambda**2/fa   # axion mass in GeV
    maHz        = ma/hbar   # axion mass in GHz
    ga = alpEM/(2.*np.pi*fa)*(2./3.)*(4. + 0.48)/1.48 # axion-photon coupling in GeV^-1
    BGeV = GaussToGeV2*Bfld # B field in GeV^2
    
    vrel0  = 1.e-3          # relative velocity in units of c
    vrel   = vrel0*cs/pc    # relative velocity in pc/s
    bandwidth0 = vrel0**2/(2.*np.pi)*maHz*1.e9 # Bandwidth in Hz
    rcT   = rc(theta, Bfld, Prd, maHz) # conversion radius in pc
    
    vc    = 0.544467*np.sqrt(RNS/rcT) # free-fall velocity at rc in units of c
    BWD   = bandwidth0*(ut/vrel)**2    # bandwidth in Hz
    
    if (profile == "PL"):
        rhoa  = MC_profile_self(density, r)
    elif (profile == "NFW"):
        rhoa = MC_profile_NFW(density, r)
        
    Flux  = np.pi/6.*ga**2*vc*(RNS/rcT)**3*BGeV**2*np.abs(3.*np.cos(theta)**2-1.)*(rhoa*RNS**3/ma)
    
    # 1.e32 is the conversion from SI to muJy. hbar converts from GeV to s^-1
    if ret_bandwidth:
        return Flux/(BWD*4.*np.pi*(s0*pc)**2*hbarT) * 1.e32, BWD
    else:
        return Flux/(BWD*4.*np.pi*(s0*pc)**2*hbarT) * 1.e32,

def n(rho, psi):
    # The AMC stars with a positions defined by rho in pc
    # Its angle out of the plane is given by psi
    rho0 =  1.4e7*1e-9 # Msun pc^-3, see Table 1 in 1304.5127
    rs   = 16.1e3      # pc
    Menc = 4*np.pi*rho0*rs**3*(np.log((rs+rho)/rs) - rho/(rs+rho))
    gravfactor = lambda t: np.sqrt(G_pc*(Menc)/rho**3)*t
    R   = lambda t: np.sqrt((np.cos(psi)*rho*np.cos(gravfactor(t)))**2 + (rho*np.sin(gravfactor(t)))**2)
    Z   = lambda t: np.sin(psi)*rho*np.cos(gravfactor(t))
    n_t = lambda t: nNS(R(t), Z(t))
    return n_t

def Ntotal(nfunc, Tage, sigmau):
    Ntfunc = lambda t: nfunc(t)*sigmau
    tlist  = np.linspace(0, Tage, 1000)
    return np.trapz(Ntfunc(tlist), x=tlist)

def Gamma(nfunc, Tage, sigmav ):
    Ntfunc = lambda t: nfunc(t)*sigmav
    tlist = np.geomspace(1, Tage, 1000)
    return np.trapz(Ntfunc(tlist), x=tlist)

def inverse_transform_sampling_log(function, x_range, nbins=1000, n_samples=1000):
    bins = np.geomspace(x_range[0], x_range[-1], num=nbins)
    pdf = function(np.delete(bins,-1) + np.diff(bins)/2)
    # Norm = np.sum(pdf*np.diff(bins))
    Norm = np.trapz(pdf, x=np.delete(bins,-1) + np.diff(bins)/2)
    pdf /= Norm
    # cum_values = np.zeros(bins.shape)
    cum_values = cumtrapz(pdf, x=np.delete(bins,-1) + np.diff(bins)/2, initial=0.0)
    inv_cdf = interp1d(cum_values, np.delete(bins,-1) + np.diff(bins)/2)
    r = np.random.rand(n_samples)
    return inv_cdf(r)

def inverse_transform_sampling(function, x_range, nbins=1000, n_samples=1000):
    bins = np.linspace(x_range[0], x_range[-1], num=nbins)
    pdf = function(np.delete(bins,-1) + np.diff(bins)/2)
    Norm = np.trapz(pdf, x=np.delete(bins,-1) + np.diff(bins)/2)
    pdf /= Norm
    # cum_values = np.zeros(bins.shape)
    cum_values = cumtrapz(pdf, x=np.delete(bins,-1) + np.diff(bins)/2, initial=0.0)
    inv_cdf = interp1d(cum_values, np.delete(bins,-1) + np.diff(bins)/2)
    r = np.random.rand(n_samples)
    return inv_cdf(r)

def dPdR(bmax, Nsamples=1000):
    # b in km
    # bmin should be the minimum impact encounter
    brange = np.array([b0,bmax])
    DF_b = lambda x: 2*b/(bmax**2 - b0**2)
    blist = inverse_transform_sampling(DF_b, brange, n_samples=Nsamples)
    return blist

def Elist(Vlist, blist, Mp, Ms, Rrms2):
    # V in km s^-1
    # M in Msun
    return 4*(G**2)*(Mp**2)*Ms*Rrms2/3/(Vlist**2)/(blist**4)
