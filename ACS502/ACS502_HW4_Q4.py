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
    r = 64.E-3
    a1 = 12.E-3
    a2 = 8.E-3
    S1 = pi*a1*a2
    a = np.sqrt(S1/pi)
    V = pi*(r**2)*L

    leff = 2*beta*a

    f0 = (c0/(2*pi))*np.sqrt(pi/(2*beta*V))

    print f0
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))