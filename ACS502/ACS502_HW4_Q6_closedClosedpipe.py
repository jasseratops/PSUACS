# PSUACS
# ACS502_HW4_Q6_closedClosedpipe
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/13/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    L = 5.
    x = np.linspace(0.,L,100)
    c = 343.
    t = 0
    plt.figure()
    for n in range(1,4):
        k = n*pi/L
        p = exp(1j*(k*c)*t)*(cos(k*x))
        plt.plot(x,p)

    plt.figure()
    for n in range(1,4):
        k = n*pi/L
        u = -k*exp(1j*(k*c)*t)*(sin(k*x))
        plt.plot(x,u)

    plt.show()


    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))