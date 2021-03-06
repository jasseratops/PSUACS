import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt


def main(args):
    a = 1.E-3
    rho = 999.
    # https://www.engineeringtoolbox.com/water-density-specific-weight-d_595.html
    p0 = 1.e5
    gamma = 1.4
    S=4.*pi*(a**2)
    c0 = 343.
    c_f= 1500.
    V = (4*pi*(a**3))/3.
    m_rad = rho*S*a
    s_gas = gamma*p0*(S**2)/V
    f = (1./(2.*pi*a))*np.sqrt((3.*gamma*p0)/(rho))
    f2 = (1/(2*pi))*np.sqrt(s_gas/m_rad)

    T= 1./f

    print f
    print f2
    print T



    omega = 2*pi*f
    k = omega/c_f
    ka = k*a
    print "ka: " + str(ka)

    Rrad = rho*c_f*S*((ka**2)/(1+(ka**2)))
    print "Rrad:" + str(Rrad)

    beta = Rrad/(2*m_rad)
    print beta

    tau = 1./beta

    print tau

    print tau/T

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))