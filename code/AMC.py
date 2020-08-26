import numpy as np
from scipy.interpolate import interp1d

from pathlib import Path
script_dir = str(Path(__file__).resolve().parent) + "/"


G_N = 4.301e-3 #(km/s)^2 pc/Msun

z_eq = 3400
rho_eq = 1512.0 #Solar masses per pc^3

#NFW properties
c = 100.
x_M_NFW,y_M_NFW = np.loadtxt(script_dir + "../data/MassLoss_NFW.txt", unpack=True)
#dM_interp_NFW = interp1d(x_M_NFW, y_M_NFW, bounds_error=False,fill_value = 0.0)
def dM_interp_NFW(x):
    return np.interp(x, x_M_NFW, y_M_NFW, left=0.0, right=1.0)

def f_NFW(x):
    return np.log(1+x) - x/(1+x)

x_E_NFW,y_E_NFW = np.loadtxt(script_dir + "../data/EnergyLoss_NFW.txt", unpack=True)
#dEloss_interp_NFW = interp1d(x_E_NFW, y_E_NFW, bounds_error=False,fill_value = 0.0)
def dEloss_interp_NFW(x):
    return np.interp(x, x_E_NFW, y_E_NFW, left=0.0, right=1.0)

#dEloss_interp_PL = interp1d(x_E_NFW, 0.0*y_E_NFW, bounds_error=False,fill_value = 0.0)
x_E_PL,y_E_PL = np.loadtxt(script_dir +"../data/EnergyLoss_PL.txt", unpack=True)
def dEloss_interp_PL(x):
    return np.interp(x, x_E_PL, y_E_PL, left=0.0, right=1.0)

#PL properties
x_M_PL,y_M_PL = np.loadtxt(script_dir +"../data/MassLoss_PL.txt", unpack=True)
#dM_interp_PL = interp1d(x_M_PL, y_M_PL, bounds_error=False,fill_value = 0.0)
def dM_interp_PL(x):
    return np.interp(x, x_M_PL, y_M_PL, left=0.0, right=1.0)


#Initialise the interpolation between rho and delta
delta_list = np.linspace(0, 1000, 10000)
rho_list = 140*(1+delta_list)*delta_list**3*(rho_eq/2.)
delta_of_rho_interp = interp1d(rho_list, delta_list)

def delta_of_rho(rho1):
    if (rho1 > 1e10):
        return ((2*rho1/rho_eq)/140)**(1/4)
    else:
        return np.interp(rho1, rho_list, delta_list)
        #return delta_of_rho_interp(rho1)


class AMC:
    
    def __init__(self, M, delta, profile="PL"):
        self.M = M
        self.delta = delta
        
        self.rho = 140*(1+delta)*delta**3*(rho_eq/2.)
        
        self.profile = profile
        
        if (profile == "PL"):
            self.k = 3/11 #Prefactor for R^2
            self.alpha = 3/2 #Prefactor for E_bind
            self.Sigma = 6/5 #Prefactor for velocity dispersion
            self.R = (3*self.M/(4*np.pi*self.rho))**(1/3.)
            self.dE_threshold = 1e-1
            self.dM_interp = dM_interp_PL
            self.dEloss_interp = dEloss_interp_PL
            
        elif (profile == "NFW"):
            self.k = 0.133
            self.alpha = 3.46
            self.Sigma = 1.02
            self.R = c*(self.M/(4*np.pi*self.rho*f_NFW(c)))**(1/3.)
            self.dE_threshold = 1e-4
            self.dM_interp = dM_interp_NFW
            self.dEloss_interp = dEloss_interp_NFW
        
        else:
            raise ValueError("AMC profile parameter must be `PL' or `NFW'") 
        
        #self.bmax = np.sqrt(self.R)*1e6
    
        
    def Ebind(self):
        return self.alpha*G_N*self.M**2/self.R
        
    def Ekinetic(self):
        return 0.5*self.Sigma*self.Ebind()
    
    def Etotal(self):
        return (0.5*self.Sigma - 1)*self.Ebind()
    
    def Rrms2(self):
        return self.k*self.R**2
        
    def rho_mean(self):
        return self.M/(4*np.pi*self.R**3/3)
    
    def disrupt(self):
        self.M = 1e-30
        self.R = 1e-30
        self.rho = 1e-30
        self.delta = 1e-30
    
    def perturb(self, dE):
        dE_frac = dE/self.Ebind()
        #print(dE_frac)
        if (dE_frac < self.dE_threshold):
            dM = 0.0
            dE_remain = dE
        else:
            dM = self.dM_interp(dE_frac)*self.M
            dE_remain = dE*(1 - self.dEloss_interp(dE_frac))
            #dE_remain = dE*(1 - dM/self.M)
        Mnew = self.M - dM
        
        #print(np.sqrt((5/3)*(self.R/G_N)*dE*(Mnew/self.M)))

        #a1 = Mnew/self.M
        #a2 = 1 - self.dEloss_interp(dE_frac)
        #d = np.abs(a1 - a2)
        #if (d > 1e-3):
        #    print(dE_frac, a1, a2, dE_frac*a2)
        #E_f = self.Etotal() + dE*(1 - self.dEloss_interp(dE_frac)) 
        #E_f = self.Etotal() + dE*(Mnew/self.M)
        E_f = self.Etotal() + dE_remain
        
        #Rnew = self.R*(Mnew**2/(self.M**2 - (5/3)*(self.R/G_N)*dE*(Mnew/self.M)))
        Rnew = (0.5*self.Sigma - 1)*self.alpha*G_N*Mnew**2/E_f
            
        #sigsq_new = (self.M/Mnew)*self.sigma_v()**2 - 2*dE/self.M
        #print(np.sqrt(sigsq_new))
        #self.R = G_N*Mnew/(2*sigsq_new)
        self.R = Rnew
            
        #The minicluster has been disrupted if R < 0
        if (self.R <= 0):
            self.disrupt()
        else:
            self.M = Mnew
            self.rho = 3*self.M/(4*np.pi*self.R**3)
            #if (self.rho > 1e20):
            #    print(dE_frac)
            if (self.profile == "NFW"): #Account for different definitions of rho
                self.rho *= c/(3*f_NFW(c))
            self.delta = delta_of_rho(self.rho)
        

    
