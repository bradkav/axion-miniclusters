#!/usr/bin/env python3

import numpy as np



#m_a = 33.086e-6
m_a = 41.564e-6
#m_a = 47.561e-6  # eV

#m_a = 8.27e-6
#m_a = 16.54e-6
#m_a = 35.16e-6

hbar  = 6.58211957e-16
print("Using mass [eV]:", m_a, f" (corresponding to {m_a*1e-9/(2*np.pi*hbar)} GHz)")

#exit()

# m_a = 306e-6 #eV ( = 74 GHz)
# m_a = 480e-6 #eV

mstr1 = str(int(np.floor(m_a*1e6)))
mstr2 = str(int(np.round((m_a*1e6 - np.floor(m_a*1e6))*1e3)))
mstr2 = mstr2.rjust(3, "0")
#
#print("CHECKING IDSTR:", IDstr2)

IDstr = "_ma_" + mstr1 + "_" + mstr2 + "mueV_M31"

#IDstr = "_ma_33_086mueV_M31"
#IDstr = "_ma_41_564mueV_M31"
#IDstr = "_ma_47_561mueV_M31"
#IDstr = "_test"

min_enhancement = 1e-2  # Threshold for 'interesting' encounters (as a fraction of the local DM density)