# PSUACS
# ACS502_HW4_Q3
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/13/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    c0 = 343.
    f = 80.
    TL = 6.
    diam1 =  0.0127
    r1 = diam1/2.

    lam = c0/f
    l = lam/8.
    k = ((2*pi)/lam)
    print "k: " +str(k)
    print "l: " + str(l)
    print "k*l: " + str(k*l)

    S1 = pi*(r1**2)

    print sin(k*l)

    a = 1.
    b = (2*S1*np.sqrt((10.**(TL/10.))-1.))/(sin(k*l))
    c = -1*(S1**2)

    roots = np.roots([a,b,c])

    print roots
    S2 = roots[1]
    print S1
    print S2
    diam2 = 2*np.sqrt(S2/pi)
    print diam2

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))