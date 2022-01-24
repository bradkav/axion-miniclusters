import numpy as np
import MilkyWay as MW
import mass_function

from scipy.interpolate import interp1d
from scipy.integrate import quad

import dirs

G    = 4.32275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun

    

def inverse_transform_sampling(function, x_range, nbins=1000, n_samples=1000, logarithmic=False):
    if (logarithmic):
        bins = np.geomspace(x_range[0], x_range[-1], num=nbins)
    else:
        bins = np.linspace(x_range[0], x_range[-1], num=nbins)
    pdf = function(np.delete(bins,-1) + np.diff(bins)/2)
    Norm = np.sum(pdf*np.diff(bins))
    pdf /= Norm
    cumul_values = np.zeros(bins.shape)
    cumul_values[1:] = np.cumsum(pdf*np.diff(bins))
    inv_cdf = interp1d(cumul_values, bins)
    r = np.random.rand(n_samples)
    return inv_cdf(r)

#Draw impact parameters
def dPdb(bmax, b0=0.0, Nsamples=1000):
    # b in km
    # bmin should be the minimum impact encounter
    
    #brange = np.array([b0,bmax])
    #DF_b = lambda b: 2*b/(bmax**2 - b0**2)
    #blist = inverse_transform_sampling(DF_b, brange, n_samples=Nsamples)

    blist = bmax*np.sqrt(np.random.uniform(size=Nsamples))
    return blist


#Calculate total number of encounters
def Ntotal(nfunc, Tage, bmax, Vp, b0=0.0):

    Ntfunc = lambda t: nfunc(t)*np.pi*Vp*(bmax**2 - b0**2) 
    tlist = np.linspace(0, Tage, 1000)
    return np.trapz(Ntfunc(tlist), x=tlist)

#Draw random samples for the encounter velocity
def dPdV(v_amc, sigma, Nsamples=1000):
    v_vec = np.atleast_2d(sigma).T*np.random.randn(Nsamples, 3)
    Vlist = np.sqrt((v_amc + v_vec[:,0])**2 + v_vec[:,1]**2 + v_vec[:,2]**2)
    #Vlist = sig_rel*np.sqrt(np.sum(v_vec**2, axis=-1))
    return Vlist

#Local circular speed
def Vcirc(Mstar, rho):
    # import astropy.units as u
    # import gala.potential as gp
    # from gala.units import galactic
    rho0 =  1.4e7*1e-9 # Msun pc^-3, see Table 1 in 1304.5127
    rs   = 16.1e3     # pc
    Menc = 4*np.pi*rho0*rs**3*(np.log((rs+rho)/rs) - rho/(rs+rho))
    # print(rho, Menc, np.sqrt(G_pc*(Mstar+Menc)/rho))
    return np.sqrt(G_pc*(Mstar+Menc)/rho) # pc/s

#Velocity dispersion at a given radius rho
def sigma(rho):
    rho0 =  1.4e7*1e-9 # Msun pc^-3, see Table 1 in 1304.5127
    rs   = 16.1e3     # pc
    Menc = 4*np.pi*rho0*rs**3*(np.log((rs+rho)/rs) - rho/(rs+rho))
    # print(rho, Menc, np.sqrt(G_pc*(Mstar+Menc)/rho))
    rho_clip = np.clip(rho, 1e-20, 1e20)
    return np.sqrt(G*(Menc)/rho_clip) # km/s


#Stellar number density in terms of galactocentric radius rho and inclination psi
def n(M, rho, psi):
    # import astropy.units as u
    # import gala.potential as gp
    # from gala.units import galactic, solarsystem, dimensionless
    # The AMC starts with a positions defined by rho in pc
    # Its angle out of the plane is given by psi
    rho0 =  1.4e7*1e-9 # Msun pc^-3, see Table 1 in 1304.5127
    rs   = 16.1e3      # pc
    
    #MW mass enclosed within radius rho
    Menc = 4*np.pi*rho0*rs**3*(np.log((rs+rho)/rs) - rho/(rs+rho)) 
    gravfactor = lambda t: np.sqrt(G_pc*(M+Menc)/rho**3)*t

    # X = \rho*cos\psi*cos\gravfactor
    # Y = \rho*sin\gravfactor
    # R = sqrt(X**2+Y**2)
    R = lambda t: np.sqrt((np.cos(psi)*rho*np.cos(gravfactor(t)))**2 + (rho*np.sin(gravfactor(t)))**2)

    Z = lambda t: np.sin(psi)*rho*np.cos(gravfactor(t))

    # n_t = lambda t: dndV_stellar(R(t), Z(t))
    n_t = lambda t: (MW.rho_star(R(t), Z(t)))/MW.M_star_avg #BJK: divided by mass of a star to get number density
    return n_t

#Calculate perturbation energy
def Elist(Vlist, blist, Mp, Ms, Rrms2):
    # V in km s^-1
    # M in Msun
    return 4*(G**2)*(Mp**2)*Ms*Rrms2/3/(Vlist**2)/(blist**4)


#--------- Functions for dealing with elliptic integrals----


def n_ecc(orb, psi):
    #T = calc_T_orb(a)

    # X = \rho*cos\psi*cos\theta
    # Y = \rho*sin\theta
    # R = sqrt(X**2+Y**2)
    
    def n_t(t):
        #r = calc_r(t, T, a, e)
        #theta = calc_theta(t, T, e)
        
        r = orb.r_of_t(t)
        theta = orb.theta_of_t(t)
        
        R = np.sqrt((r*np.cos(theta)*np.cos(psi))**2 + (r*np.sin(theta))**2)
        Z = r*np.cos(theta)*np.sin(psi)
        return MW.rho_star(R, Z)/MW.M_star_avg
    
    # R = lambda t: np.sqrt((calc_r(t, T, a, e)*np.cos(calc_theta(t, T, e))*np.cos(psi))**2
                                #  + (calc_r(t, T, a, e)*np.sin(calc_theta(t, T, e)))**2)
    # Z = lambda t: calc_r(t, T, a, e)*np.cos(calc_theta(t, T, e))*np.sin(psi)
    
    # n_t = lambda t: (MW.rho_star(R(t), Z(t)))/MW.M_star_avg #BJK: divided by mass of a star to get number density
    return n_t

#BJK: Note that Vp becomes a function of time...
def Ntotal_ecc(Tage, bmax, orb, psi, b0=0.0):
    #M = calc_M_enc(a)
    #T = calc_T_orb(a)
    #M = orb.M_enc

    nfunc = n_ecc(orb, psi)
    Vp = lambda t: orb.vis_viva_t(t)
    Ntfunc = lambda t: nfunc(t)*np.pi*Vp(t)*(bmax**2 - b0**2) 
    tlist = np.linspace(0, orb.T_orb, 1000)

    N_orb = np.trapz(Ntfunc(tlist), x=tlist)

    return N_orb*(Tage/orb.T_orb)

def sample_ecc(N):
    elist_loaded, P_e_loaded = np.loadtxt(dirs.data_dir + 'eccentricity.txt', unpack=True, delimiter=',')
    P_e = interp1d(elist_loaded, P_e_loaded, bounds_error=False, fill_value = 0.0)
    erange = np.linspace(0,1,100)
    return inverse_transform_sampling(P_e, erange, n_samples=N)

def dPdVamc(orb, psi, bmax, Nsamples, b0=0.0):
    
    #M = calc_M_enc(a)
    #T = calc_T_orb(a)
    nfunc = n_ecc(orb, psi)
    Vp = lambda t: orb.vis_viva_t(t)
    Ntfunc = lambda t: nfunc(t)*np.pi*Vp(t)*(bmax**2 - b0**2)

    tlist = inverse_transform_sampling(Ntfunc, np.linspace(0,orb.T_orb/2,10), n_samples=Nsamples)
    #rlist = calc_r(tlist, T, a, e)
    rlist = orb.r_of_t(tlist)

    # tlist = np.random.uniform(0,T_orb, Nsamples)
    return orb.vis_viva_t(tlist) * 3.08567758e13, rlist # pc s^-1 to km s^-1
