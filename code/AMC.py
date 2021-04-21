import numpy as np
from scipy.interpolate import interp1d

from pathlib import Path
script_dir = str(Path(__file__).resolve().parent) + "/"


G_N = 4.301e-3 #(km/s)^2 pc/Msun

z_eq = 3400
rho_eq = 1512.0 #Solar masses per pc^3

def f_NFW(x):
    return np.log(1+x) - x/(1+x)

#NFW properties
c = 100.
dE_NFW, dM_NFW, fej_NFW, fub_NFW = np.loadtxt(script_dir + "../data/Perturbations_NFW.txt", unpack=True)
#dM_interp_NFW = interp1d(x_M_NFW, y_M_NFW, bounds_error=False,fill_value = 0.0)
def dM_interp_NFW(x):
    return np.interp(x, dE_NFW, dM_NFW, left=0.0, right=1.0)
    
def fej_interp_NFW(x):
    return np.interp(x, dE_NFW, fej_NFW, left=0.0, right=1.0)
    
def fub_interp_NFW(x):
    return np.interp(x, dE_NFW, fub_NFW, left=0.0, right=1.0)

c_alt = 1000.
dE_NFWd, dM_NFWd, fej_NFWd, fub_NFWd = np.loadtxt(script_dir + "../data/Perturbations_NFWd.txt", unpack=True)
#dM_interp_NFW = interp1d(x_M_NFW, y_M_NFW, bounds_error=False,fill_value = 0.0)
def dM_interp_NFWd(x):
    return np.interp(x, dE_NFWd, dM_NFWd, left=0.0, right=1.0)
    
def fej_interp_NFWd(x):
    return np.interp(x, dE_NFWd, fej_NFWd, left=0.0, right=1.0)
    
def fub_interp_NFWd(x):
    return np.interp(x, dE_NFWd, fub_NFWd, left=0.0, right=1.0)
    

c10000 = 10000.
dE_NFWc10000, dM_NFWc10000, fej_NFWc10000, fub_NFWc10000 = np.loadtxt(script_dir + "../data/Perturbations_NFW_c10000.txt", unpack=True)
#dM_interp_NFW = interp1d(x_M_NFW, y_M_NFW, bounds_error=False,fill_value = 0.0)
def dM_interp_NFWc10000(x):
    return np.interp(x, dE_NFWc10000, dM_NFWc10000, left=0.0, right=1.0)
    
def fej_interp_NFWc10000(x):
    return np.interp(x, dE_NFWc10000, fej_NFWc10000, left=0.0, right=1.0)
    
def fub_interp_NFWc10000(x):
    return np.interp(x, dE_NFWc10000, fub_NFWc10000, left=0.0, right=1.0)
    
dE_PL, dM_PL, fej_PL, fub_PL = np.loadtxt(script_dir + "../data/Perturbations_PL.txt", unpack=True)
#dEloss_interp_NFW = interp1d(x_E_NFW, y_E_NFW, bounds_error=False,fill_value = 0.0)
def dM_interp_PL(x):
    return np.interp(x, dE_PL, dM_PL, left=0.0, right=1.0)
    
def fej_interp_PL(x):
    return np.interp(x, dE_PL, fej_PL, left=0.0, right=1.0)
    
def fub_interp_PL(x):
    return np.interp(x, dE_PL, fub_PL, left=0.0, right=1.0)


#Initialise the interpolation between rho and delta
delta_list = np.linspace(0, 1000, 10000)
rho_list = 140*(1+delta_list)*delta_list**3*(rho_eq/2.)
delta_of_rho_interp = interp1d(rho_list, delta_list)
rho_of_delta_interp = interp1d(delta_list, rho_list)
print("%.3e, %.3e"%(rho_of_delta_interp(0.1), rho_of_delta_interp(20)))
print("%.3e"%(rho_of_delta_interp(1),))
print(delta_of_rho_interp(1e6))

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
            self.alpha_sq = 3/11 #Prefactor for R^2
            self.beta = 3/2 #Prefactor for E_bind
            self.kappa = 1.73 #Prefactor for velocity dispersion
            self.R = (3*self.M/(4*np.pi*self.rho))**(1/3.)
            self.dE_threshold = 1e-4
            self.dM_interp = dM_interp_PL
            self.fej_interp = fej_interp_PL
            self.fub_interp = fub_interp_PL
            
        elif (profile == "NFW"):
            self.alpha_sq = 0.133 #Prefactor for R^2
            self.beta = 3.47 #Prefactor for E_bind
            self.kappa = 3.54 #Prefactor for velocity dispersion
            self.R = c*(self.M/(4*np.pi*self.rho*f_NFW(c)))**(1/3.)
            self.dE_threshold = 1e-4
            self.dM_interp = dM_interp_NFW
            self.fej_interp = fej_interp_NFW
            self.fub_interp = fub_interp_NFW
        
        elif (profile == "NFWd"):
            self.alpha_sq = 0.084 #Prefactor for R^2
            self.beta = 14.167 #Prefactor for E_bind
            self.kappa = 14.208 #Prefactor for velocity dispersion
            rho_s = self.rho/0.58
            self.R = c_alt*(self.M/(4*np.pi*rho_s*f_NFW(c_alt)))**(1/3.)
            self.dE_threshold = 1e-5
            self.dM_interp = dM_interp_NFWd
            self.fej_interp = fej_interp_NFWd
            self.fub_interp = fub_interp_NFWd
            
        elif (profile == "NFWc10000"):
            self.alpha_sq = 0.063 #Prefactor for R^2
            self.beta = 74.29 #Prefactor for E_bind
            self.kappa = 74.32 #Prefactor for velocity dispersion
            #rho_s = self.rho/0.58
            rho_s = self.rho
            self.R = c10000*(self.M/(4*np.pi*rho_s*f_NFW(c10000)))**(1/3.)
            self.dE_threshold = 1e-6
            self.dM_interp = dM_interp_NFWc10000
            self.fej_interp = fej_interp_NFWc10000
            self.fub_interp = fub_interp_NFWc10000
        
        else:
            raise ValueError("AMC profile parameter must be `PL' or `NFW'") 
        
        #self.bmax = np.sqrt(self.R)*1e6
    
        
    def Ebind(self):
        return self.beta*G_N*self.M**2/self.R
        
    def Ekinetic(self):
        return 0.5*self.kappa*G_N*self.M**2/self.R
    
    def Etotal(self):
        return (0.5*self.kappa/self.beta - 1)*self.Ebind()
    
    def Rrms2(self):
        return self.alpha_sq*self.R**2
        
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
            dE_remain = dE*(1 - self.fej_interp(dE_frac)) - self.fub_interp(dE_frac)*self.Etotal()
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
        
        if (E_f >= 0):
            self.disrupt()
        else:
        
            #Rnew = self.R*(Mnew**2/(self.M**2 - (5/3)*(self.R/G_N)*dE*(Mnew/self.M)))
            Rnew = (0.5*self.kappa - self.beta)*G_N*Mnew**2/E_f
            
            #sigsq_new = (self.M/Mnew)*self.sigma_v()**2 - 2*dE/self.M
            #print(np.sqrt(sigsq_new))
            #self.R = G_N*Mnew/(2*sigsq_new)
            self.R = Rnew
            self.M = Mnew
            self.rho = 3*self.M/(4*np.pi*self.R**3)
            #if (self.rho > 1e20):
            #    print(dE_frac)
            if (self.profile == "NFW"): #Account for different definitions of rho
                self.rho *= c/(3*f_NFW(c))
            if (self.profile == "NFWd"):
                self.rho *= 0.58*c_alt/(3*f_NFW(c_alt))
            if (self.profile == "NFWc10000"):
                self.rho *= c10000/(3*f_NFW(c10000))
            self.delta = delta_of_rho(self.rho)
        

    
