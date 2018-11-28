# PSUACS
# ACS502_HW6_Q5
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/15/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    SPL = 73
    prms = (20.E-6)*(10**(73/20))
    rho = 1.21
    c = 343.
    r = 10.
    z = r*c
    Qk =2*np.sqrt(2)*r*prms/(rho*c)
    P = (rho*c)*(Qk**2)/(4*pi)

    print P

    r = (8/2)*(z/np.sqrt(2))*(Qk/2*pi)/((20.E-6)*(10**(100/20)))
    print r



    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))