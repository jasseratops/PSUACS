# PSUACS
# ACS_HW4_Q8
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/13/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    rho0 = 1.21
    c0 = 343.

    Z0 = rho0*c0
    SWR = 1.7

    R = - (SWR-1)*(exp(1j*pi/2))/(SWR+1)
    print np.abs(R)
    print np.angle(R)
    Zn = Z0*((1+R)/(1-R))

    print Zn
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))