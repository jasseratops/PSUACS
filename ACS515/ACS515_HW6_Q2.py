# PSUACS
# ACS515_HW6_Q2
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/19/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    deg = 0.01
    degRes = deg/360.
    theta = np.pi * np.arange(0,2,degRes)
    R_array = np.array([0.1,2.5,10])
    k=1.

    for i in R_array:
        plt.figure()
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, quad_dir(theta,i,k), label="R="+str(i))
        plt.legend()
        ax.grid(True)
        plt.savefig("ACS515_HW6_Q2_FIG_R_"+str(i).replace(".","p"))
    plt.show()

    return 0

def quad_dir(theta,R,k):
    i= -1j
    dir = ((1.-(3.*((cos(theta))**2)))*((i*k/R)-(1./(R**2.))+((k**2.)/3.)))-((k**2.)/3.)
    return np.abs(dir)

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))