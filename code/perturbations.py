import numpy as np
import MilkyWay as MW
import mass_function
G    = 4.32275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun
from scipy.interpolate import interp1d
from scipy.integrate import quad


    
def sample_AMCs_logflat(m_a = 2e-5, n_samples=1000):
        #First sample the masses                                                                                                                               
    #It turns out that in this case, we can do the inverse                                                                                                 
    #sampling analytically if we have a power law distribution                                                                                            
    #print(M_max(m_a))                                                                                                                                    
    x_list = np.random.uniform(np.log(mass_function.calc_Mmin(m_a)), np.log(mass_function.calc_Mmax(m_a)), size = n_samples)
    #M1 = M_min(m_a)**(1+gamma)                                                                                                                           
    #M2 = M_max(m_a)**(1+gamma)                                                                                                                            
    #M_list = (x_list*(M2 - M1) + M1)**(1/(1+gamma))                                                                                                      
    M_list = np.exp(x_list)
    #Now sample delta                                                                                                                                      

    delta_list = inverse_transform_sampling(mass_function.P_delta, [0.1, 20], \
                       nbins=1000, n_samples=n_samples)
    return M_list, delta_list
    

def inverse_transform_sampling(function, x_range, nbins=1000, n_samples=1000):
    bins = np.linspace(x_range[0], x_range[-1], num=nbins)
    pdf = function(np.delete(bins,-1) + np.diff(bins)/2)
    Norm = np.sum(pdf*np.diff(bins))
    pdf /= Norm
    cum_values = np.zeros(bins.shape)
    cum_values[1:] = np.cumsum(pdf*np.diff(bins))
    inv_cdf = interp1d(cum_values, bins)
    r = np.random.rand(n_samples)
    return inv_cdf(r)

def dPdb(bmax, b0=0.0, Nsamples=1000):
    # b in km
    # bmin should be the minimum impact encounter
    
    #brange = np.array([b0,bmax])
    #DF_b = lambda b: 2*b/(bmax**2 - b0**2)
    #blist = inverse_transform_sampling(DF_b, brange, n_samples=Nsamples)

    blist = bmax*np.sqrt(np.random.uniform(size=Nsamples))
    return blist


#BJK: Note that Vp becomes a function of time...
def Ntotal(nfunc, Tage, bmax, Vp, b0=0.0):

    Ntfunc = lambda t: nfunc(t)*np.pi*Vp*(bmax**2 - b0**2) 
    #tlist = np.geomspace(1, Tage, 1000) #BJK: Why is this geomspace, rather than linspace?
    tlist = np.linspace(0, Tage, 1000)
    # import matplotlib.pyplot as plt
    # plt.semilogy(tlist,nfunc(tlist))
    # plt.show()
    # print(Ntfunc(tlist))
    # quit()
    return np.trapz(Ntfunc(tlist), x=tlist)

#This looks like the dispersion is sig_rel/np.sqrt(2)
#It's also very slow...
def dPdV_old(sig_rel, Nsamples=1000):
    # V in km s^-1
    # Vmax set to 10 times the value of the virial velocity
    # if not vpeak:
    Vrange = np.array([0,300*10])
    DF_v = lambda V: (4*np.pi*V**2*np.exp(-V**2/sig_rel**2))/(2*np.pi*sig_rel**2)**(3/2)
    Vlist = inverse_transform_sampling(DF_v, Vrange, n_samples=Nsamples)
    return Vlist
    
    
def dPdV_old2(sig_rel, Nsamples=1000):
    # V in km s^-1
    # Vmax set to 10 times the value of the virial velocity
    # if not vpeak:
    #Vrange = np.array([0,300*10])
    #DF_v = lambda V: (4*np.pi*V**2*np.exp(-V**2/sig_rel**2))/(2*np.pi*sig_rel**2)**(3/2)
    v_vec = np.random.randn(Nsamples, 3)
    Vlist = sig_rel*np.sqrt(np.sum(v_vec**2, axis=-1))
    #Vlist = inverse_transform_sampling(DF_v, Vrange, n_samples=Nsamples)
    return Vlist
    # if vpeak:
        # return 2*np.sqrt(sig_rel)*np.sqrt(2/np.pi)

def dPdV(v_amc, sigma, Nsamples=1000):
    v_vec = np.atleast_2d(sigma).T*np.random.randn(Nsamples, 3)
    Vlist = np.sqrt((v_amc + v_vec[:,0])**2 + v_vec[:,1]**2 + v_vec[:,2]**2)
    #Vlist = sig_rel*np.sqrt(np.sum(v_vec**2, axis=-1))
    return Vlist

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

#LV: Not used?
def dndV_stellar(R, Z):
    # returns in Msun pc^-3
    Rsun = 8.3e3 # pc
    Zsun = 0.0 # pc
    nnorm_disk = 0.1 # Msun pc^-3
    nnorm_halo = 5.25e-5 # Msun pc^-3
    nnorm_bulge = 400. # Msun pc^-3
    hR = 2.3e3 # pc
    hz = 0.82e3 # pc
    qh = 0.637
    nH = 2.59
    n_disc = lambda R_t, Z_t: nnorm_disk*np.exp(Rsun/hR)*np.exp((-R_t/hR) - (Z_t-Zsun)/hz)
    n_halo = lambda R_t, Z_t: nnorm_halo*(Rsun/np.sqrt(R_t**2 + (Z_t/qh)**2))**nH
    # n_bulge = lambda R_t, Z_t: nnorm_bulge*np.exp()/(1 + a/a0)
    return n_disc(R,Z) + n_halo(R,Z) #+ n_bulge(R,Z)


#BJK: This needs to be extended to be in terms of (a, e, psi), rather than just rho, the galactocentric distance
#Or in terms of (R, z, then feed in the coordinates as a function of t)
#Note also that the disruption rate depends on v 
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

def nLV(M, rho, psi):
    rho0 =  1.4e7*1e-9 # Msun pc^-3, see Table 1 in 1304.5127
    rs   = 16.1e3      # pc
    Menc = 4*np.pi*rho0*rs**3*(np.log((rs+rho)/rs) - rho/(rs+rho))
    t = 4.26e17
    Deltat=t/1000
    gravfactor = np.sqrt(G_pc*(M+Menc)/rho**3)*t
    Vcirc      = np.sqrt(G_pc*(M+Menc)/rho)
    R = np.sqrt((np.cos(psi)*rho*np.cos(gravfactor))**2 + (rho*np.sin(gravfactor))**2)
    Z = np.sin(psi)*rho*np.cos(gravfactor)
    n_t = MW.rho_star(R, Z)
    bmax = 1e-3 
    N_t = n_t * np.pi*Vcirc*bmax**2*Deltat
    return N_t

def Elist(Vlist, blist, Mp, Ms, Rrms2):
    # V in km s^-1
    # M in Msun
    return 4*(G**2)*(Mp**2)*Ms*Rrms2/3/(Vlist**2)/(blist**4)


# def calc_orbit(M, a, e):
#     # M in Msun
#     # a in kpc
#     # e is eccentricity
#     T_orb = (2*np.pi)*np.sqrt(a**3/(G_N*M))
#     t_list = np.linspace(0,T_orb,10000)
#     r_list = calc_r(t_list, T_orb, a, e)
#     theta_list = calc_theta(t_list, T_orb, e)
#     return r_list, theta_list, t_list

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

    # import matplotlib.pyplot as plt
    # plt.semilogy(calc_r(tlist, T, a, e),nfunc(tlist), label='e = %.2f, a = %.2f'% (e,a,))
    # plt.legend()
    # plt.xlabel('r')
    # plt.ylabel('dN/dr')
    # plt.savefig('../plots/dNdr.pdf', bbox_inches='tight')
    # quit()
    #print(bmax)
    #print(Vp(tlist))
    #print(nfunc(tlist))

    N_orb = np.trapz(Ntfunc(tlist), x=tlist)

    return N_orb*(Tage/orb.T_orb)

def sample_ecc(N):
    elist_loaded, P_e_loaded = np.loadtxt('eccentricity.txt', unpack=True, delimiter=',')
    P_e = interp1d(elist_loaded, P_e_loaded, bounds_error=False, fill_value = 0.0)
    erange = np.linspace(0,1,100)
    return inverse_transform_sampling(P_e, erange, n_samples=N)

def dPdVamc(orb, psi, bmax, Nsamples, b0=0.0):
    
    #M = calc_M_enc(a)
    #T = calc_T_orb(a)
    nfunc = n_ecc(orb, psi)
    Vp = lambda t: orb.vis_viva_t(t)
    Ntfunc = lambda t: nfunc(t)*np.pi*Vp(t)*(bmax**2 - b0**2)
    # import matplotlib.pyplot as plt
    # tlist = np.linspace(0,T/2,100)
    # plt.semilogy(tlist,Ntfunc(tlist))
    # plt.show()
    # quit()

    tlist = inverse_transform_sampling(Ntfunc, np.linspace(0,orb.T_orb/2,10), n_samples=Nsamples)
    #rlist = calc_r(tlist, T, a, e)
    rlist = orb.r_of_t(tlist)
    # import matplotlib.pyplot as plt
    # tlist = np.linspace(0,T/2,100)
    # plt.hist(calc_r(tlist, T, a, e)/a,)
    # plt.show()
    # quit()
    # print(calc_r(tlist, T, a, e)/a)
    # quit()


    # tlist = np.random.uniform(0,T_orb, Nsamples)
    return orb.vis_viva_t(tlist) * 3.08567758e13, rlist # pc s^-1 to km s^-1
