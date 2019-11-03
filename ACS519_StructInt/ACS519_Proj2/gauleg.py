# PSUACS
# gauleg
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/3/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    X,Y = gauleg(0,pi/2,4)
    print "X: " + str(X)
    print "Y: " + str(Y)
    return 0

def gauleg(X1,X2,N):
    eps = 3e-14
    M = (N+1)/2
    print M
    XM = 0.5*(X2+X1)
    XL = 0.5*(X2-X1)

    X = np.zeros(N)
    W = np.zeros(N)

    for i in range(M):
        i+=1
        print "i+1: " + str(i)
        Z = cos(pi*(i-0.25)/(N+0.5))
        Z1 = -100000

        while np.abs(Z-Z1) > eps:
            P1 = 1
            P2 = 0
            for j in range(N):
                j+=1
                P3 = P2
                P2 = P1
                P1 = ((2*j-1)*Z*P2-(j-1)*P3)/j
            PP = N*(Z*P1-P2)/(Z*Z-1)
            Z1 = Z
            Z = Z1-P1/PP
        i-=1
        X[i] = XM-XL*Z
        X[-i-1] = XM+XL*Z
        W[i]= 2*XL/((1-Z**2)*PP**2)
        W[-i-1] = W[i]

    return X,W

if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))