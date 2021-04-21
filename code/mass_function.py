import numpy as np
G    = 4.32275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import quad
from abc import ABC, abstractmethod, abstractproperty

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

#Calculate the mass of a minicluster of tidal stripping from the MW:
#Sec. 2.2 of https://arxiv.org/abs/1403.6827
A_MW = 1.34
zeta = 0.07
t_dyn = 2.4e9
t_MW = 13.5e9
M_MW = 1e12
def mass_after_stripping(m_i):
    return m_i*(1 + zeta*(m_i/M_MW)**zeta*(A_MW*t_MW/t_dyn))**(-1/zeta)


class GenericMassFunction(ABC):
    
    #These 3 parameters - mmin, mmax, mavg - are essential
    #and have to appear in all MassFunction classes
    #self.mmin = 0
    #self.mmax = 1e30
    #self.mavg = 1e30
    
    @abstractmethod
    def dPdlogM_internal(self, mass):
        """
        Edit this halo mass function, dP/dlogM
        Strongly recommend making this vectorized
        """
        pass
    
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

        result = 0.0*mass
        
        inds = (self.mmin < mass) & (mass < self.mmax)
        
        result[inds] = self.dPdlogM_internal(mass[inds])
        
        #result[mass > self.mmax] = 0.
        #result[mass < self.mmin] = 0.
        
        if scalar_input:
            return (np.squeeze(result)).item(0)
        return result
        
        
    def calc_norm(self):
        m_list = np.geomspace(self.mmin, self.mmax, 2000)
        P_list = self.dPdlogM(m_list)
        return np.trapz(P_list/m_list, m_list)
    


#-------------------------------------------------------------------
class PowerLawMassFunction(GenericMassFunction):
    
    def __init__(self, m_a, gamma):
        
        #These parameters are specific to the model we use
        self.gamma = gamma
        self.m_a = m_a
        
        self.mmin = calc_Mmin(m_a)
        self.mmax = calc_Mmax(m_a)
        
        #Here, we generally need the average mass *before* any disruption, so let's calculate this
        #before we do any correction for stripping due to the MW halo
        self.mavg = ((gamma)/(gamma + 1))*(self.mmax**(gamma + 1) - self.mmin**(gamma+1))/(self.mmax**gamma - self.mmin**gamma)    
        
    def dPdlogM_internal(self, mass):
        """
        Edit this halo mass function, dP/dlogM
        Strongly recommend making this vectorized
        """
        return self.gamma*mass**self.gamma/(self.mmax**self.gamma-self.mmin**self.gamma)

        
    

#------------------------------------------------------------------
class StrippedPowerLawMassFunction(GenericMassFunction):
    
    def __init__(self, m_a, gamma):
        
        #These parameters are specific to the model we use
        self.gamma = gamma
        self.m_a = m_a
        
        #Here 'us' denotes 'unstripped', i.e. the values before MW stripping has been accounted for
        self.mmin_us = calc_Mmin(m_a)
        self.mmax_us = calc_Mmax(m_a)
        
        #Here, we generally need the average mass *before* any disruption, so let's calculate this
        #before we do any correction for stripping due to the MW halo
        self.mavg = ((gamma)/(gamma + 1))*(self.mmax_us**(gamma + 1) - self.mmin_us**(gamma+1))/(self.mmax_us**gamma - self.mmin_us**gamma)
        
        mi_list = np.geomspace(self.mmin_us, self.mmax_us, 10000)
        mf_list = mass_after_stripping(mi_list)
        
        self.mmin = np.min(mf_list)
        self.mmax = np.max(mf_list)
        print("M_max:", self.mmax)
        
        self.mi_of_mf = InterpolatedUnivariateSpline(mf_list, mi_list, k=1, ext=1)
        self.dmi_by_dmf = self.mi_of_mf.derivative(n=1)
        
    def dPdlogM_nostripping(self, mass):
        return self.gamma*mass**self.gamma/(self.mmax_us**self.gamma-self.mmin_us**self.gamma)
        
        
    def dPdlogM_internal(self, mass):
        """
        Edit this halo mass function, dP/dlogM
        Strongly recommend making this vectorized
        """
        m_f = mass
        m_i = self.mi_of_mf(m_f)
        
        return self.dPdlogM_nostripping(m_i)*self.dmi_by_dmf(m_f)*m_f/m_i
