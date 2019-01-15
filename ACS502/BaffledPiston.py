import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt


def main(args):

    ka = np.linspace(0.005,50,1024)
    '''Z_rad = 
    '''
    SPL = 140.
    pref = 20.E-6
    dP = pref*(10**(SPL/20.))
    P = 1.E5
    gam = 1.401
    T = 288.

    dT = ((gam-1)/(gam))*(T/P)*(dP)
    print dT

    MW_N2 = 28.0134
    MW_O2 = 31.998
    MW_Ar = 39.948
    MW_CO2 = 44.01

    per_N2 = 78.09E-2
    per_O2 = 20.95E-2
    per_Ar = 00.93E-2
    per_CO2= 00.04E-2

    gam_N2 = 1.4
    gam_O2 = 1.4
    gam_Ar = 1.667
    gam_CO2= 1.333

    MW = MW_N2*per_N2+MW_O2*per_O2+MW_Ar*per_Ar+MW_CO2*per_CO2
    print MW
    MW *= 1.E-3

    gam = gam_N2*per_N2+gam_O2*per_O2+gam_Ar*per_Ar+gam_CO2*per_CO2
    print gam
    R = 8.314
    T = 293.


    c = np.sqrt(gam*R*T/MW)
    print c

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))