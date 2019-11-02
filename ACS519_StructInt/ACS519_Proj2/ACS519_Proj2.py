# PSUACS
# ACS519_Proj1
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/2/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    c = 343.
    f = np.linspace(0,10.E3,1024)
    omega = 2.*pi*f
    k = omega/c

    a = 1.
    b = 3.

    phi = np.linspace(0,pi/2.,1024)
    theta = np.linspace(0,pi/2.,1024)

    phi,theta = np.meshgrid(phi, theta)

    alpha = k*a*sin(theta)*cos(phi)
    beta  = k*b*sin(theta)*sin(phi)


    return 0


def gauleg(X1,X2,N):
    eps = 3e-14
    M = (N+1)/2
    XM = 0.5*(X2+X1)
    XL = 0.5*(X2-X1)

    X = np.zeros(M)
    W = np.zeros(M)

    for i in range(M):
        i+=1
        Z = cos(pi*(i-0.25)/(N+0.5))
        Z1 = -100000

        while abs(Z-Z1) > eps:
            P1 = 1
            P2 = 0
            for j in range(N):
                j+1
                P3 = P2
                P2 = P1
                P1 = ((2*j-1)*Z*P2-(j-1)*P3)/j
            PP = N*(Z*P1-P2)/(Z*Z-1)
            Z1 = Z
            Z = Z1-P1/PP

        i-=1
        X[i] = XM-XL*Z
        X[N+1-i] = XM+XL*Z
        W[i]= 2*XL/((1-Z**2)*PP**2)
        W[N+1-i] = W[i]

    return X,W

if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))