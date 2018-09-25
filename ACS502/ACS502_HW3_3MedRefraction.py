# PSUACS
# ACS502_HW3_3MedRefraction
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/24/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def twoMed(rhoA,cA,rhoB,cB,theta_i):
    zA = rhoA*cA
    zB = rhoB*cB

    theta_tr = np.arcsin((cB/cA)*sin(theta_i))
    theta_cr = np.arcsin(cA/cB)

    R = ((zB*cos(theta_i))-(zA*cos(theta_tr)))/((zB*cos(theta_i))+(zA*cos(theta_tr)))

    return R, theta_cr

def pMatGen(theta_i,Yacs1,Yacs2,Yacs3,k1l,k2l,k3l):
    length = len(theta_i)
    P = np.zeros(length,dtype=complex)


    for i in range(length):
        print i

        M = np.array([[                       1.,                       1.,      -1.,                         0],
                      [                 Yacs2[i],                -Yacs2[i], Yacs1[i],                         0],
                      [          exp(-1j*k2l[i]),           exp(1j*k2l[i]),        0,          -exp(-1j*k3l[i])],
                      [ exp(-1j*k2l[i])*Yacs2[i], -exp(1j*k2l[i])*Yacs2[i],        0, -exp(-1j*k3l[i])*Yacs3[i]]])
        print M

        c = np.array([1.,Yacs1[i],0.,0.])

        invM = np.linalg.inv(M)

        P[i] = np.matmul(invM,c)
        print P


    print P

    return P


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


    length = 1024

    theta_i = np.linspace(0.,pi/2.,length)
    theta_gr = (pi/2.) - theta_i
    theta_t_13 = np.arcsin((c3/c1)*sin(theta_i))


    R_13, theta_13_cr = twoMed(rhoA=rho1,cA=c1,rhoB=rho3,cB=c3,theta_i=theta_i)


    plt.figure()
    plt.plot(np.degrees(theta_gr),abs(R_13))

    plt.figure()
    plt.plot(np.degrees(theta_gr),abs(R_13))

    theta_2 = np.arcsin((c2/c1)*sin(theta_i))
    theta_tr = np.arcsin((c3/c2)*sin(theta_2))
    Yacs1 = cos(theta_i)/z1
    Yacs2 = cos(theta_2)/z2
    Yacs3 = cos(theta_tr)/z3

    kl = [0.2, 2., 20.]

    plt.figure()
    for i in kl:
        k1l = i
        print k1l
        k2l = k1l * (sin(theta_i) / sin(theta_2))
        k3l = k1l * (sin(theta_i) / sin(theta_tr))
        R3med = pMatGen(theta_i, Yacs1, Yacs2, Yacs3, i, k2l, k3l)
        plt.plot(np.degrees(theta_gr),R3med,label="kl= " +str(i))

    plt.legend()
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))