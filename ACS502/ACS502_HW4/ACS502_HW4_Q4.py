# PSUACS
# ACS502_HW4_Q4
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/13/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    beta = 0.785
    c0 = 343.
    L = 122.E-3
    r = (64.E-3)/2
    a1 = 12.E-3
    a2 = 8.E-3
    S = pi*a1*a2
    a = np.sqrt(S/pi)
    V = pi*(r**2)*L
    l0 = 0.
    #corr = 2*beta
    corr = 2*(8/(3.*pi))
    leff = l0 + corr*a


    f0 = (c0/(2*pi))*np.sqrt(S/(leff*V))

    print f0
    k = 2*pi*f0/c0
    print k*L

    f2 = (np.arctan(a/r))*c0/(2*pi*leff)

    print f2

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))