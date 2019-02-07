import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt


def main(args):
    a = 1.E-3
    rho = 999.06
    # https://www.engineeringtoolbox.com/water-density-specific-weight-d_595.html
    p0 = 100.E3
    gamma = 1.4
    S=4.*pi*(a**2)
    c0 = 343.
    V = (4*pi*(a**3))/3.

    m_a = 4.*pi*(a**3.)*3./(3.*(1.+(rho*(a**2.))/(gamma*p0)))
    c_a = V/(gamma*p0)
    f = (1./(2.*pi))*np.sqrt((3.*gamma*p0)/(rho*(a**2)))
    f2= (1./(2.*pi))*np.sqrt(1./(c_a*m_a))
    T= 1./f
    print f
    print f2
    print T



    omega = 2*pi*f
    k = omega/c0
    ka = k*a

    Rrad = rho*c0*S*((ka**2)/(1+(ka**2)))
    beta = Rrad / 2*m_a





    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))