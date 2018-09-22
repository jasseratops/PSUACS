# PSUACS
# ACS502_SonicBoom_Fish
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/22/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    A = 95.70
    p_rms = A/np.sqrt(3)

    pRef_water = 1.E-6
    pRef_air = 20.E-6


    rho_air=1.21
    rho_water = 1026.0
    c_air = 343.0
    c_water = 1500.0

    z_air = rho_air*c_air
    z_water = rho_water*c_water

    T = (2*z_water)/(z_water + z_air)

    A_tr = T*A
    p_tr = T*p_rms
    SPL_tr = 20*np.log10(p_tr/pRef_water)

    tau = (T**2)*(z_air/z_water)
    I_tr = (p_tr**2)/z_water
    TL = -10*np.log10(tau)

    print "p_rms: " + str(p_rms)
    print "z_air: " + str(z_air)
    print "z_water: " + str(z_water)
    print "T: " + str(T)
    print "A_tr: " + str(A_tr)
    print "p_tr: " + str(p_tr)
    print "SPL_tr: " + str(SPL_tr)
    print "tau: " + str(tau)
    print "I_tr: " + str(I_tr)
    print "TL: " + str(TL)

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))