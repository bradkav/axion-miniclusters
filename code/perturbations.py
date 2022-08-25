import numpy as np
import mass_function

from scipy.interpolate import interp1d
from scipy.integrate import quad

import dirs
import tools

G    = 4.32275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun

    
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



#Calculate perturbation energy
def Elist(Vlist, blist, Mp, Ms, Rrms2):
    # V in km s^-1
    # M in Msun
    return 4*(G**2)*(Mp**2)*Ms*Rrms2/3/(Vlist**2)/(blist**4)


#--------- Functions for dealing with elliptic integrals----


def n_ecc(orb, psi, galaxy):
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
        return galaxy.rho_star(R, Z)/galaxy.M_star_avg
    
    # R = lambda t: np.sqrt((calc_r(t, T, a, e)*np.cos(calc_theta(t, T, e))*np.cos(psi))**2
                                #  + (calc_r(t, T, a, e)*np.sin(calc_theta(t, T, e)))**2)
    # Z = lambda t: calc_r(t, T, a, e)*np.cos(calc_theta(t, T, e))*np.sin(psi)
    
    # n_t = lambda t: (MW.rho_star(R(t), Z(t)))/MW.M_star_avg #BJK: divided by mass of a star to get number density
    return n_t

#BJK: Note that Vp becomes a function of time...
def Ntotal_ecc(Tage, bmax, orb, psi, galaxy, b0=0.0):
    #M = calc_M_enc(a)
    #T = calc_T_orb(a)
    #M = orb.M_enc

    nfunc = n_ecc(orb, psi, galaxy)
    Vp = lambda t: orb.vis_viva_t(t)
    Ntfunc = lambda t: nfunc(t)*np.pi*Vp(t)*(bmax**2 - b0**2) 
    tlist = np.linspace(0, orb.T_orb, 1000)

    N_orb = np.trapz(Ntfunc(tlist), x=tlist)

    #print(orb.T_orb)
    return N_orb*(Tage/orb.T_orb)

def sample_ecc(N):
    elist_loaded, P_e_loaded = np.loadtxt(dirs.data_dir + 'eccentricity.txt', unpack=True, delimiter=',')
    P_e = interp1d(elist_loaded, P_e_loaded, bounds_error=False, fill_value = 0.0)
    erange = np.linspace(0,1,100)
    return tools.inverse_transform_sampling(P_e, erange, n_samples=N)

def dPdVamc(orb, psi, bmax, Nsamples, galaxy, b0=0.0):
    
    #M = calc_M_enc(a)
    #T = calc_T_orb(a)
    nfunc = n_ecc(orb, psi, galaxy)
    Vp = lambda t: orb.vis_viva_t(t)
    Ntfunc = lambda t: nfunc(t)*np.pi*Vp(t)*(bmax**2 - b0**2)

    tlist = tools.inverse_transform_sampling(Ntfunc, np.linspace(0,orb.T_orb/2,10), n_samples=Nsamples)
    #rlist = calc_r(tlist, T, a, e)
    rlist = orb.r_of_t(tlist)

    # tlist = np.random.uniform(0,T_orb, Nsamples)
    return orb.vis_viva_t(tlist) * 3.08567758e13, rlist # pc s^-1 to km s^-1
