import numpy as np
G    = 4.32275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun
from scipy.interpolate import interp1d
from scipy.integrate import quad

#def SampleAMC(n_samples):
    
def P_delta(delta):
    #The overdensity distribution df/d\[Delta] is defined
    #in Eq. (S39) in https://arxiv.org/abs/1906.00967.
    #Here, we've corrected a few small errors.
    sigma = 0.448
    n = 11.5
    deltaG = 1.06
    S = 4.7
    d = 1.93
    alpha = -0.21
    deltaF = 3.4
    A = 1/2.045304
    
    B1 = np.exp(-(np.abs(alpha)/np.sqrt(2))**d)
    B2 = ((np.sqrt(2)/np.abs(alpha))**d*np.abs(alpha)*n/d)
    C = np.abs(alpha)*((np.sqrt(2)/np.abs(alpha))**d*n/d + 1)
    
    Pdelta = np.zeros(delta.shape)
    
    x = np.log(delta/deltaG)
    
    mask1 = (x <= sigma*alpha)
    mask2 = (x > sigma*alpha)
    
    Pdelta[mask1] = np.exp(-(np.abs(np.abs(x[mask1]))/(np.sqrt(2)*sigma))**d)
    Pdelta[mask2] = B1*(C/B2 + x[mask2]/(sigma*B2))**-n
    
    return Pdelta*A/(1 + (delta/deltaF)**S)

def calc_Mmin(m_a):
    #Minimum AMC mass in Msun
    #m_a - axion mass in eV

    #These values are valid if you neglect axion stars
    
    # MJeans is Eq.B18 in 1707.03310
    # M_min is Eq.23 in 1707.03310 at z=0
    
    
    MJeans = 5.1e-10*(m_a/1e-10)**(-3/2)
    M_min = MJeans*(1.8/7.5)**2
    
    
    #This value is a cut for possible problems with axion stars
    #M_min = 3e-16
    return M_min
    
def calc_Mmax(m_a):
    #Maximum AMC mass in Msun
    #m_a - axion mass in eV  

    # M0 is found in Eq.34 in 1808.01879 
    M0 = 6.6e-12*(m_a/5e-5)**(-1/2)

    # M_max is Eq.22 in 1707.03310 at z=0
    return 4.9e6*M0

class PowerLawMassFunction:
    
    def __init__(self, m_a, gamma):
        
        #These parameters are specific to the model we use
        self.gamma = gamma
        self.m_a = m_a
        
        #These 3 parameters - mmin, mmax, mavg - are essential
        #and have to appear in all MassFunction classes
        #I could probably use abstract base classes and things
        #here but that stuff tends to put people off...
        self.mmin = calc_Mmin(m_a)
        self.mmax = calc_Mmax(m_a)
        
        self.mavg = ((gamma)/(gamma + 1))*(self.mmax**(gamma + 1) - self.mmin**(gamma+1))/(self.mmax**gamma - self.mmin**gamma)
        
    def dPdlogM_internal(self, mass):
        """
        Edit this halo mass function, dP/dlogM
        Strongly recommend making this vectorized
        """
        return self.gamma*mass**self.gamma/(self.mmax**self.gamma-self.mmin**self.gamma)


    def dPdlogM(self, mass):
        """
        This wrapper function ensures that the HMF is zero 
        outside of mmin < mass < mmax and also ensures that
        it evaluates correctly for both scalar and vector 'mass'
        """
        mass = np.asarray(mass)
        scalar_input = False
        if mass.ndim == 0:
            mass = mass[None]  # Makes x 1D
            scalar_input = True

        result = self.dPdlogM_internal(mass)
        
        result[mass > self.mmax] = 0.
        result[mass < self.mmin] = 0.
        
        if scalar_input:
            return (np.squeeze(result)).item(0)
        return result
        
    def calc_norm(self):
        m_list = np.geomspace(self.mmin, self.mmax, 2000)
        P_list = self.HMF(m_list)
        return np.trapz(P_list/m_list, m_list)