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
    omega1 = 2*pi*f1
    omega2 = 2 * pi * f2
    k1 = 2*pi*f1/c
    k2 = 2 * pi * f2 / c
    S = pi*(a**2)
    print k1*a
    print k2*a
    m = 0.070

    xi1 = (1/(a*f1))*np.sqrt(W/(2*rho*c*(pi**3)))
    xi2 = (1/(a*f2))*np.sqrt(W/(2*rho*c*(pi**3)))

    xi1= (1/omega1)*np.sqrt((2*W)/(rho*c*S))
    xi12= (1/omega2)*np.sqrt((2*W)/(rho*c*S))

    Mr1 = pi*(a**2)*rho*(8/(3*pi))*a

    Xr2 = (pi*(a**2)*rho*c)/(2*pi*k2*a)
    Mr2 = Xr2/omega2

    print xi1
    print xi2

    print "Mr1: " + str(Mr1)
    Mtot1 = m + Mr1
    per1 = Mr1/Mtot1
    print "per1: " + str(per1)

    print "Mr2: " + str(Mr2)
    Mtot2 = m + Mr2
    per2 = Mr2 / Mtot2
    print "per2: " + str(per2)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))