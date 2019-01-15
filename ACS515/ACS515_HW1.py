# PSUACS
# ACS515_HW1
# Jasser Alshehri
# Starkey Hearing Technologies
# 1/15/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):

    x = np.linspace(-3.,3.,1024)
    t = 0
    rho0 = 1.21
    c0 = 343.
    P0 = 1.01325E5
    T0 = 293.
    gam = 1.4

    Phi = exp(-((x-(343.*t))**2))
    u = -2*(x-(c0*t))*Phi
    p = -rho0*2*c0*(x-(c0*t))*Phi
    rho = -(rho0/c0)*2*(x-(c0*t))*Phi
    T = -rho0*((gam-1)/gam)*(T0/P0)*2*c0*(x-(c0*t))*Phi


    plt.figure()
    plt.plot(x,Phi)
    plt.title("Velocity Potential")
    plt.xlabel("Displacement [m]")
    plt.ylabel("Velocity Potential")

    plt.figure()
    plt.subplot(411)
    plt.plot(x, u)
    plt.subplot(412)
    plt.plot(x, p)
    plt.subplot(413)
    plt.plot(x, rho)
    plt.subplot(414)
    plt.plot(x, T)
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))