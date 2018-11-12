import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys


def main(args):

    pr = np.sqrt(2.)*2.
    r = 5.
    r0 = 0.01
    rho0 = 1.21
    c0 = 343.
    '''
    rho0 = 1026.
    c0 = 1500.
    '''
    f = 500.
    omega = 2*pi*f


    A = (pr*r)/(omega*rho0*r0)

    k = omega/c0
    u0 = np.sqrt(((A/r0)**2)+((k*A)**2))

    print A
    print u0

    u0_2 = -A*(1j*k + (1/r0))

    print np.abs(u0)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))