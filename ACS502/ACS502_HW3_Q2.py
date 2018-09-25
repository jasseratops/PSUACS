# PSUACS
# ACS502_HW3_Q2
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/25/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    f =18.E6

    rho1 = 1000.
    c1 = 1500.
    z1 = rho1*c1
    print z1

    rho3 = 7860.
    Y3 = 205.E9
    c3 = np.sqrt(Y3/rho3)
    z3 = rho3*c3
    print z3

    z2 = np.sqrt(z1*z3)
    n = 600.
    k2 = (n-0.5)*pi*(1./2.5)*(1./0.0254)

    c2 = 2*pi*f/k2

    rho2 = z2/c2
    print c3
    print z2
    print k2
    print c2
    print rho2

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))