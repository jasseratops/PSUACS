# PSUACS
# ACS515_HW5_Q2
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/14/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    rho0 = 1.21
    c = 343.
    P_avg = 1.
    f = 50.
    omega = 2*pi*f
    k = omega/c
    diam = 0.2
    a = diam/2.
    i=-1j

    ka= k*a

    u_s = np.sqrt((2*P_avg*(1+(ka**2)))/(rho0*c*(4*pi*(a**2))*(ka**2)))
    print "u_s: " + str(u_s)
    xi = u_s/omega
    print "Lambda: " + str(c/f)
    print "ka: " + str(ka)
    print "xi: " + str(xi)
    S = -i*omega*rho0*(a**2)*u_s
    print "S: " + str(S)
    Q = (-4*pi)*S/(i*omega*rho0)
    print "Q: " + str(Q)

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))