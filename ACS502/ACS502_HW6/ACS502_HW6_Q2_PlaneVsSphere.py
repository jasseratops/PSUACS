import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys


def main(args):
    rho0 = 1.21
    c0 = 343.
    Z0 = rho0*c0
    f0 = 7000.
    omega = 2*pi*f0
    k = omega/c0

    p_rms = 14.16
    u_rms = p_rms/Z0
    xi_rms = u_rms/omega

    print p_rms
    print u_rms
    print xi_rms

    r = 3.E-3
    A = r*np.sqrt(2)*p_rms
    print A

    u = (1-(1j/(k*r)))*(1/(rho0*c0))*(A/r)
    u_rms = np.abs(u)/np.sqrt(2)
    print np.abs(u)
    print u_rms

    xi_rms = u_rms/omega

    print xi_rms

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))