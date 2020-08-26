import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
G_N = 6.67408e-11*6.7702543e-20 # pc^3 solar mass^-1 s^-2 (conversion: m^3 kg^-1 s^-2 to pc^3 solar mass^-1 s^-2)

def calc_M_enc(a):
    rho0 =  1.4e7*1e-9 # Msun pc^-3, see Table 1 in 1304.5127
    rs   = 16.1e3      # pc
    
    #MW mass enclosed within radius a
    Menc = 4*np.pi*rho0*rs**3*(np.log((rs+a)/rs) - a/(rs+a)) 
    return Menc

#def calc_T_orb(a):
#    Menc = calc_M_enc(a) 
#    T_orb = (2*np.pi)*np.sqrt(a**3/(G_N*Menc))
#    return T_orb

class elliptic_orbit:
    
    def __init__(self, a, e):
        self.a = a
        self.e = e
        
        
        self.M_enc = calc_M_enc(a)
        self.T_orb = (2*np.pi)*np.sqrt(a**3/(G_N*self.M_enc))
    
        #Initialise interpolation for eccentric anomaly
        self.E_anom_list = np.linspace(0, 2*np.pi, 1000)
        self.M_anom_list = self.E_anom_list - self.e*np.sin(self.E_anom_list)
        self.E_anom_interp = interp1d(self.M_anom_list, self.E_anom_list, bounds_error=False, fill_value = 0.0)
    
        #Initialise interpolation functions for (r, theta) as a function of t
        self.t_list     = np.linspace(0, self.T_orb, 1000)
        self.r_list     = self.calc_r(self.t_list)
        self.theta_list = self.calc_theta(self.t_list)
        
        self.r_of_t     = interp1d(self.t_list, self.r_list)
        self.theta_of_t = interp1d(self.t_list, self.theta_list)
        
# --------- Functions for solving elliptical orbits        

    def vis_viva_t(self, t):
        #r = calc_r(t, self.T_orb, a, e)
        r = self.r_of_t(t)
        return ((G_N*self.M_enc)*(2/r - 1/self.a))**0.5

    def vis_viva_r(self, r):
        return ((G_N*self.M_enc)*(2/r - 1/self.a))**0.5
        
    
    def calc_M_anom(self,t):
        #M = mean anomaly
        frac = (t%self.T_orb)/self.T_orb #M should be between 0 and 2pi
        return (2 * np.pi * frac)
    

    def calc_E(self,M_anom):
        # M = mean anomaly
        # E = eccentric anomaly
        # e = eccentricity
        m = lambda E: M_anom - E + (self.e * np.sin(E))
        Elist = np.linspace(0,2*np.pi)
        return brentq(m, 0, 2*np.pi)

    def calc_E_interp(self,M_anom):
        # M = mean anomaly
        # E = eccentric anomaly
        # e = eccentricity
        return self.E_anom_interp(M_anom)

    #@np.vectorize
    def calc_theta(self, t):
        # (1 - e)tan^2(theta/2) = (1 + e)tan^2(E/2)
        # e = eccentricity
        # theta = true anomaly
        # E = eccentric anomaly
        M_anom = self.calc_M_anom(t)
        E = self.calc_E_interp(M_anom)
        #theta_func = lambda theta: (1 - self.e) * np.tan(theta/2)**2 - (1 + self.e) * np.tan(E/2)**2
        #theta = brentq(theta_func, 0, np.pi)
        arg = np.sqrt(((1 + self.e)/(1 - self.e)) * np.tan(E/2)**2)
        theta =  2*np.arctan(arg)
    
        mask = t >= self.T_orb/2
        theta[mask] = 2*(np.pi - theta[mask]) + theta[mask]
        return theta
        #if t < self.T_orb/2:
        #    return theta
        #else:
        #    return 2*(np.pi - theta) + theta
    
    #@np.vectorize
    def calc_r(self, t):
        # a = semi-major axis
        # r = a(1 - ecosE)
        M_anom = self.calc_M_anom(t)
        E = self.calc_E_interp(M_anom)
        return self.a * (1 - (self.e * np.cos(E)))
    
    








