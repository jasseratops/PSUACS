# PSUACS
# Q7
# Jasser Alshehri
# Starkey Hearing Technologies
# 12/6/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    a = 2.5E-3
    f = 3.E3
    d = (3.E-6)/2.
    c = 343.
    rho = 1.21
    SPL_0p2 = 71.
    r = 0.2

    lam = c/f
    print "lam: " + str(lam)
    omega = 2*pi*f
    k = omega/c
    print "ka: " + str(k*a)

    xi = 100.E-6
    U = xi*omega
    Q = U*pi*(a**2)
    print "Q: " + str(Q)

    Rr = pi*(a**2)*rho*c*((k*a)**2)/2.

    W_M = (0.5)*(U**2)*Rr

    Q2 = (20.E-6)*(10**(SPL_0p2/20.))*(np.sqrt(2))*4*pi*r/\
        (rho*k*c*2*k*d)
    print Q2

    W_D = rho*c*(Q2**2)*(k**4)*(d**2)/(6*pi)
    W_baff = rho*c*(Q**2)*(k**2)/(4*pi)

    print "W_M: " + str(W_M)
    print "W_D: " + str(W_D)
    print "W_baff: "+str(W_baff)
    print W_baff/W_D


    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))