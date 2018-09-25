# PSUACS
# ACS502_HW3_3MedRefraction
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/24/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    '''
    rho1= 1025.
    c1 = 1450.
    Z1 = rho1*c1

    rho2= 1600.
    c2 = 1550.
    Z2 = rho2 * c2

    rho3 = 1650.
    c3 = 1600.
    Z3 = rho3 * c3
    '''

    #brewster
    rho1= 1020.
    c1 = 1520.
    Z1 = rho1*c1

    rho2= 1600.
    c2 = 1550.
    Z2 = rho2 * c2

    rho3 = 1400.
    c3 = 1500.
    Z3 = rho3 * c3

    f= 1200.
    omega = f*2*pi

    theta_i = np.linspace(0.,pi/2.,1024*4)
    print np.degrees(theta_i)
    theta_gr = (pi/2.) - theta_i
    print np.degrees(theta_gr)
    theta_t_13 = np.arcsin((c3/c1)*sin(theta_i))


    R_13 = ((Z3*cos(theta_i))-(Z1*cos(theta_t_13)))/((Z3*cos(theta_i))+(Z1*cos(theta_t_13)))

    theta_cr_13 = np.arcsin(c1/c3)
    print "Theta_cr_13: " + str(np.degrees(theta_cr_13))


    plt.figure()
    plt.plot(np.degrees(theta_gr),abs(R_13))

    plt.figure()
    plt.plot(np.degrees(theta_i),abs(R_13))
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))