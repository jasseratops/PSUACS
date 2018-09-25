# PSUACS
# ACS502_HW3_3MedRefraction
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/24/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):

    rho1= 1025.
    c1 = 1450.
    z1 = rho1*c1

    rho2= 1600.
    c2 = 1550.
    z2 = rho2 * c2

    rho3 = 1650.
    c3 = 1600.
    z3 = rho3 * c3


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
    '''
    f= 1200.
    omega = f*2*pi

    length = 1024

    theta_i = np.linspace(0.,pi/2.,length)
    print np.degrees(theta_i)
    theta_gr = (pi/2.) - theta_i
    print np.degrees(theta_gr)
    theta_t_13 = np.arcsin((c3/c1)*sin(theta_i))



    R_13 = ((z3*cos(theta_i))-(z1*cos(theta_t_13)))/((z3*cos(theta_i))+(z1*cos(theta_t_13)))

    theta_cr_13 = np.arcsin(c1/c3)
    print "Theta_cr_13: " + str(np.degrees(theta_cr_13))

    theta_2 = np.arcsin((c2/c1)*sin(theta_i))
    theta_tr = np.arcsin((c3/c2)*sin(theta_2))
    Yacs1 = cos(theta_i)/z1
    Yacs2 = cos(theta_2)/z2
    Yacs3 = cos(theta_tr)/z3

    kl = [0.2, 2., 20]
    k1l = kl[0]
    k2l = k1l*(sin(theta_i)/sin(theta_2))
    k3l = k1l*(sin(theta_i)/sin(theta_tr))


    for i in range(length):

        M = np.array([[                 1.,                 1.,    -1.,                   0],
                      [              Yacs2[i],             -Yacs2[i],  Yacs1[i],                   0],
                      [       exp(-1j*k2l[i]),        exp(1j*k2l[i]),         0,       -exp(-1j*k3l[i])],
                      [ exp(-1j*k2l)*Yacs2, -exp(1j*k2l)*Yacs2,      0, exp(-1j*k3l)*Yacs3]])

        c = np.array([1.,Yacs1,0,0])

    print "passed"

    invM = np.linalg.inv(M)

    P = np.matmul(invM,c)
    print P

    R = P[2]

    plt.figure()
    plt.plot(np.degrees(theta_gr),abs(R_13))

    plt.figure()
    plt.plot(np.degrees(theta_i),abs(R_13))
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))