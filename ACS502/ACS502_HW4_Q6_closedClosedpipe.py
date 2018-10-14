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

    rho0 = 1.21
    Z0 = rho0*c
    plt.figure()
    for n in range(1,4):
        k = n*pi/L
        p = np.abs(exp(1j*(k*c)*t)*(cos(k*x)))
        plt.plot(x,p)
    plt.xlim(0,L)
    plt.ylim(0,1)
    plt.title("Pressure, Closed-Closed")
    plt.xlabel("x-position")
    plt.ylabel("|P/P+|")

    plt.figure()
    for n in range(1,4):
        k = n*pi/L
        omega= k*c
        u = np.abs((-1j/Z0)*exp(1j*(k*c)*t)*(sin(k*x)))
        plt.plot(x,u/(max(u)))
    plt.xlim(0,L)
    plt.ylim(0,1)
    plt.title("Particle Velocity, Closed-Closed")
    plt.xlabel("x-position")
    plt.ylabel("|U/U+|")
    plt.show()


    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))