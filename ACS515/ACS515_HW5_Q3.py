# PSUACS
# ACS515_HW5_Q3
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/14/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    d = 0.1
    beta = np.array([2,4,6])
    c = 343.

    f = (beta*c)/(2*pi*d)
    print f
    deg = 0.01
    degRes = deg/360.
    theta = np.pi * np.arange(0,2,degRes)

    #ax = plt.subplot(111, projection='polar')
    for i in beta:
        plt.figure()
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, Dir(i, theta)[0], label="kd="+str(i))
        ax.plot(theta, Dir(i, theta)[1])
        plt.legend()
        ax.grid(True)
        plt.savefig("kd"+str(i))
    plt.show()

    return 0

def Dir(kd,theta):
    H = cos((kd)*sin(theta))
    H_pos = np.zeros(len(H))
    H_neg = np.zeros(len(H))

    for i in range(len(H)):
        if H[i] > 0:
            H_pos[i] = H[i]
            H_neg[i] = 0
        else:
            H_pos[i] = 0
            H_neg[i] = np.abs(H[i])

    return H_pos,H_neg

if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))