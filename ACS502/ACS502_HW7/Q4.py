import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt
from scipy.special import jv

def main(args):
    c = 1500.
    rho = 1026.

    SPL = 200.
    r = 500.
    ref = 1.E-6
    f = 15.E3
    omega = 2*pi*f
    k = omega/c
    W_max = 500.

    p_req_rms = ref*(10**(SPL/20.))
    print p_req_rms

    Q = np.sqrt((8*pi*c*W_max)/((omega**2)*rho))
    print Q
    p_rms_max = (rho*c*Q*k)/(4*pi*r)
    print p_rms_max

    a = 11

    R1 = 1.-(jv(1,(2*k*a)))/(2*k*a)
    P0 = (2.*np.sqrt(2.)*(r)*(p_req_rms))/(k*(a**2))
    print P0
    W = ((pi*(a**2))/(2*rho*c))*(P0**2)*R1
    print W
    print k*a

    two_theta_HP = 2*np.arcsin(1.616/(k*a))
    print "beamwidth: " + str(np.degrees(two_theta_HP))
    R0 = (k*(a**2))/2.
    print R0

    a1 = np.sqrt(2)*1*pi/k
    a2 = np.sqrt(2) * 3 * pi / k

    print a1
    print a2

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))