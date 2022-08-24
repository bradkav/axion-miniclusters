import numpy as np
G    = 4.32275e-3       # (km/s)^2 pc/Msun
G_pc = G*1.05026504e-27 # (pc/s)^2 pc/Msun
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import quad
from abc import ABC, abstractmethod, abstractproperty

import perturbations
import params

rho_eq = 1512.0 #Solar masses per pc^3


from matplotlib import pyplot as plt

def f_NFW(x):
    return np.log(1+x) - x/(1+x)


        
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
    
def rho_of_delta(delta):
    return 140*(1+delta)*delta**3*(rho_eq/2.)
    
    #140
    
#Initialise the interpolation between rho and delta
delta_list = np.linspace(0, 1000, 10000)
rho_list = rho_of_delta(delta_list)
delta_of_rho_interp = interp1d(rho_list, delta_list)
rho_of_delta_interp = interp1d(delta_list, rho_list)

def calc_Mchar(m_a):
    return 6.6e-12*(m_a/5e-5)**(-1/2)

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
    #M0 = 6.6e-12*(m_a/5e-5)**(-1/2)
    M0 = calc_Mchar(m_a)

    # M_max is Eq.22 in 1707.03310 at z=0
    return 4.9e6*M0


def get_mass_function(ID, m_a, profile, Nbins_mass=300, unperturbed=False):
    if (ID in ["powerlaw"]):
        # Mass function
        if profile == "PL":
            AMC_MF = PowerLawMassFunction(m_a=m_a, gamma=params.gamma, profile=profile)
        elif profile == "NFW":
            print("> Warning: Ignoring stripping due to global tides!")
            #BJK: May need to implement global tidal stripping...
            AMC_MF = PowerLawMassFunction(m_a=m_a, gamma=params.gamma, profile=profile)
            #AMC_MF = mass_function.StrippedPowerLawMassFunction(m_a=in_maeV, gamma=in_gg)
        M0 = AMC_MF.mavg
    
    
    elif (ID in ["delta_a"]):
        MF_test = PowerLawMassFunction(m_a=m_a, gamma=params.gamma, profile=profile)
        M0 = MF_test.mavg
    elif (ID in ["delta_c"]):
        M0 = calc_Mchar(in_maeV)
    elif (ID in ["delta_p"]):
        M0 = 1e-14*(m_a/50e-6)**(-0.5)
    else:
        raise ValueError("Invalid mass_fuction.")
        
    if (ID in ["delta_a", "delta_c", "delta_p"]):
        AMC_MF = DeltaMassFunction(m_a=m_a, M0=M0, Nbins_mass=Nbins_mass)
    
    return AMC_MF, M0

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
    
    def dPdlogMdrho(self, mass, rho):
        """
        AMC distribution function dP/dlogMdrho = P(logM, rho)
        """
        pass
    
    def dPdlogM_internal(self, mass):
        """
        Halo mass function, dP/dlogM
        Strongly recommend making this vectorized
        """
        pass
        
    def dPdrho(self, rho):
        """
        Density distribution of AMCs (marginalised over AMC masses)
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
        
    def calc_mavg(self):
        m_list = np.geomspace(self.mmin, self.mmax, 2000)
        P_list = self.dPdlogM(m_list)
        return np.trapz(P_list, m_list)
        
        
    #Sample logflat masses for the AMCs
    def sample_AMCs_logflat(self, n_samples=1000):
        #First sample the masses                                                                                                                          
        
        #Extend an order of magnitude above and below M_min, M_max, just in case we have to
        #adjust these values later
        x_list = np.random.uniform(np.log(0.1*self.mmin), np.log(10.0*self.mmax), size = n_samples)                                                                                                  
        M_list = np.exp(x_list)                                                                                                                              

        #Then draw a sample of densities
        rho_bar_list = perturbations.inverse_transform_sampling(self.dPdrho, [self.rhomin, self.rhomax], \
                           nbins=1000, n_samples=n_samples)
    
        return M_list, rho_bar_list
        
    #Sample logflat masses for the AMCs
    def sample_AMCs(self, n_samples=1000):
        #First sample the masses                                                                                                                               

        #First, draw the densities of the AMCs
        rho_bar_list = perturbations.inverse_transform_sampling(self.dPdrho, [self.rhomin, self.rhomax], \
                           nbins=10000, n_samples=n_samples, logarithmic=True)

        #Then draw the masses from the marginal distribution
        M_list = np.zeros(n_samples)
        for i in range(n_samples):
            pdf = lambda M: self.dPdlogMdrho(M, rho_bar_list[i])/M
            M_list[i] = perturbations.inverse_transform_sampling(pdf, [self.mmin, self.mmax], \
                           nbins=10000, n_samples=1, logarithmic=True)
        

        return M_list, rho_bar_list
    


#-------------------------------------------------------------------
class PowerLawMassFunction(GenericMassFunction):
    
    def __init__(self, m_a, gamma, profile="PL", Nbins_mass=300):
        
        #These parameters are specific to the model we use
        self.gamma = gamma
        self.m_a = m_a
        
        self.mmin = calc_Mmin(m_a)
        self.mmax = calc_Mmax(m_a)
        
        self.type = "extended"
        
        #Here, we generally need the average mass *before* any disruption, so let's calculate this
        #before we do any correction for stripping due to the MW halo
        self.mavg = ((gamma)/(gamma + 1))*(self.mmax**(gamma + 1) - self.mmin**(gamma+1))/(self.mmax**gamma - self.mmin**gamma)
        self.mass_edges = np.geomspace(1e-6 * self.mmin, self.mmax, num=Nbins_mass + 1)
        
        #BJK: Deal here with the different definitions of rho for different profiles! 
        self.density_conversion = 1.0 #= <rho>/rho_AMC
        if (profile == "NFW"):
            c = 100
            self.density_conversion = (3*f_NFW(c)/c**3)
        
        if (profile == "NFWd"):
            c = 1000
            self.density_conversion = (3*f_NFW(c)/c**3)*(1.0/0.58)
        
        if (profile == "NFWc10000"):
            c = 10000
            self.density_conversion = (3*f_NFW(c)/c**3)
            
        self.rhomin = rho_of_delta(0.1)*self.density_conversion
        self.rhomax = rho_of_delta(20.0)*self.density_conversion
        
        
    def dPdlogM_internal(self, mass):
        """
        Edit this halo mass function, dP/dlogM
        Strongly recommend making this vectorized
        """
        return self.gamma*mass**self.gamma/(self.mmax**self.gamma-self.mmin**self.gamma)

    def dPdrho(self, rho):
        """
        Edit this (initial) density distribution, dP/drho
        """
        delta = delta_of_rho_interp(rho/self.density_conversion)
        drhoddelta = 140*(rho_eq/2.0)*(delta**3 + 3*(1+delta)*delta**2)
        return P_delta(delta)/drhoddelta/self.density_conversion
        
        
    def dPdlogMdrho(self, mass, rho):
        """
        Edit this joint PDF P(logM,rho)
        """
        return self.dPdlogM(mass)*self.dPdrho(rho)
        
    
class DeltaMassFunction(GenericMassFunction):
    
    def __init__(self, m_a, M0, Nbins_mass=300):
        
        
        #These parameters are specific to the model we use
        self.m_a = m_a
        self.M0 = M0
        
        self.mmin = calc_Mmin(m_a)
        self.mmax = 10*M0
        
        #This tells the code that it should deal with this
        #mass function differently, because it's a delta-function
        self.type = "delta"
        
        self.mass_edges = np.geomspace(1e-6 * self.mmin, self.mmax, num=(Nbins_mass + 1))
        self.i0 = np.digitize(M0, self.mass_edges)
        self.deltam = self.mass_edges[self.i0 + 1] - self.mass_edges[self.i0]
        self.deltalogm = np.log(self.mass_edges[self.i0 + 1]) - np.log(self.mass_edges[self.i0])
        
        #Here, we generally need the average mass *before* any disruption, so let's calculate this
        #before we do any correction for stripping due to the MW halo
        self.mavg = M0
            
        self.rhomin = rho_of_delta(1)
        self.rhomax = rho_of_delta(3)
        
        
    def dPdlogM_internal(self, mass):
        """
        Edit this halo mass function, dP/dlogM
        Strongly recommend making this vectorized
        """
        res = 0.0 * mass
        res[np.digitize(mass, self.mass_edges) == self.i0] = 1.0 / self.deltalogm
        return res

    def dPdrho(self, rho):
        """
        Edit this (initial) density distribution, dP/drho
        """
        #Let's use a flat distribution in delta, for simplicity
        delta_min = 1
        delta_max = 3
        delta = delta_of_rho_interp(rho)
        inds = (delta > delta_min) & (delta < delta_max) 
        P = (1/(delta_max - delta_min))*inds #Uniform distribution
        drhoddelta = 140*(rho_eq/2.0)*(delta**3 + 3*(1+delta)*delta**2)
        return P/drhoddelta #dP/drho = dP/ddelta/(drho/ddelta)
        
        
    def dPdlogMdrho(self, mass, rho):
        """
        Edit this joint PDF P(logM,rho)
        """
        return self.dPdlogM(mass)*self.dPdrho(rho)

#--------------------------------
        
class ExampleMassFunction(GenericMassFunction):
    
    def __init__(self):
        
        #We might in general define some parameters specific to the model we're using, 
        #but here we don't have any...
        #self.gamma = gamma
        #self.m_a = m_a
        
        self.mmin = 1e-15 #Msun
        self.mmax = 1e-5 #Msun
        
        self.rhomin = 1e1 #Msun/pc**3 
        self.rhomax = 1e5 #Msun/pc**3
        
        mlist = np.geomspace(self.mmin, self.mmax, 1000)
        rholist = np.linspace(self.rhomin, self.rhomax, 500)
        
        mgrid, rhogrid = np.meshgrid(mlist, rholist)
        dPdlogMdrho_grid = self.dPdlogMdrho(mgrid, rhogrid)
        dPdlogM_list = np.trapz(dPdlogMdrho_grid, rholist, axis=0)
        #print(dPdlogM_list.shape)
        self.dPdlogM_interp = interp1d(mlist, dPdlogM_list, bounds_error=False, fill_value = 0.0)
    
        """
        for rho1 in rholist:
            print(np.trapz(self.dPdlogMdrho(mlist, rho1)/(self.dPdrho(rho1)*mlist), mlist))
    
        print(np.trapz(self.dPdrho(rholist), rholist))
    
        print("Checking norm (1):", np.trapz(dPdlogM_list/mlist, mlist))
        print("Checking norm (2):", self.calc_norm())
        plt.figure()
        
        plt.xscale('log')
        plt.yscale('log')
        plt.contourf(rhogrid, mgrid, dPdlogMdrho_grid/mgrid)
        plt.plot(rholist, 1e-10*(rholist/1e3), 'k--')
        
        plt.figure()
        
        plt.loglog(mlist, dPdlogM_list/mlist)
        
        plt.show()
        """
        
        self.mavg = self.calc_mavg()
        
        
        
    def dPdlogM_internal(self, mass):
        """
        Edit this halo mass function, dP/dlogM
        Strongly recommend making this vectorized
        """
        return self.dPdlogM_interp(mass)

    def dPdrho(self, rho):
        """
        Edit this (initial) density distribution, dP/drho
        """
        return rho*0.0 + 1/(self.rhomax - self.rhomin)
        
    def dPdlogMdrho(self, mass, rho):
        """
        Edit this joint PDF P(logM,rho)
        """
        sigma = 1e-10
        M0 = 1e-10*(rho/1e3)
        return (2*np.pi*sigma**2)**-0.5*mass*np.exp(-0.5*(mass-M0)**2/sigma**2)*self.dPdrho(rho)
        
    

#------------------------------------------------------------------
