import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt


def main(args):
    a = 0.175
    W = 0.8
    c = 343.
    rho = 1.21
    f1 = 30.
    f2 = 7000.

    k1 = 2*pi*f1/c
    k2 = 2 * pi * f2 / c

    print k1*a
    print k2*a

    xi1 = (1/(a*f1))*np.sqrt(W/(2*rho*c*(pi**3)))
    xi2 = (1/(a*f2))*np.sqrt(W/(2*rho*c*(pi**3)))

    print xi1
    print xi2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))