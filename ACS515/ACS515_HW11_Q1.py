# PSUACS
# ACS515_HW11_Q1
# Jasser Alshehri
# Starkey Hearing Technologies
# 4/3/2019

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    a = 2.5E-2
    r = 8.
    f = 100.
    c = 343.
    SPL = 95.
    p_ref = 20.E-6
    rho = 1.21

    omega = 2*pi*f
    k = omega/c

    ka = k*a
    kr = k*r

    print "ka: " + str(ka)
    print "kr: " + str(kr)
    print "1/ka: " + str(1/ka)
    print 10*"-"

    p_pk = p_ref*10**(SPL/20.)

    print "p_pk: " + str(p_pk)
    print 10*"-"
    u0_rad_approx = (p_pk/(rho*c))*np.sqrt(r/a)*np.sqrt(2./(pi*ka))
    u0_rad_comp = (p_pk/(rho*c))*np.sqrt(pi/2.)*np.sqrt(kr)*np.sqrt(((ka/2.)**2)+((2./(pi*ka))**2))
    xi0_rad_approx = u0_rad_approx/omega

    print "Radial"
    print "Vel. (Approx.): "+str(u0_rad_approx)
    print "Vel. (Comp.): " +str(u0_rad_comp)
    print 10*"-"
    Hder = 0.5*(0.5*(2.-(ka/2)**2)-1j*(1./pi)*((2*(np.log(kr)-0.116))+(2./ka)**2))
    HderMag = np.abs(Hder)

    u0_trans_approx = (p_pk/(rho*c))*(1./ka)*np.sqrt(r/a)*np.sqrt(2./(pi*ka))
    u0_trans_comp = (p_pk/(rho*c))*np.sqrt(pi*kr/2.)*HderMag
    xi0_trans_approx = u0_trans_approx/omega

    print "Trans."
    print "Vel. (Approx.): "+str(u0_trans_approx)
    print "Vel. (Comp.): " +str(u0_trans_comp)
    print 10*"-"

    print "Displacement"
    print "Radial: " + str(xi0_rad_approx)
    print "Trans.: " + str(xi0_trans_approx)
    print "rat. trans/rad: " + str(xi0_trans_approx/xi0_rad_approx)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))