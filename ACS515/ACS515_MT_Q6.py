import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt


def main(args):
    a = 0.1E-2
    rho = 999.06
    # https://www.engineeringtoolbox.com/water-density-specific-weight-d_595.htmlg
    p0 = 1e5
    gamma = 1.4
    S=4.*pi*(a**2)
    c_f= 1498.
    f = (1./(2.*pi*a))*np.sqrt((3.*gamma*p0)/(rho))
    u_s = 0.01*a

    print f


    omega = 2*pi*f
    k = omega/c_f
    ka = k*a
    print "ka: " + str(ka)

    Rrad = rho*c_f*S*((ka**2)/(1+(ka**2)))
    print "Rrad:" + str(Rrad)

    P_rad = 0.5*(u_s**2)*Rrad
    print "P_rad: " + str(P_rad)


    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))