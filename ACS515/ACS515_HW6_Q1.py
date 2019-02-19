# PSUACS
# ACS515_HW6_Q1
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/18/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    kd = np.linspace(0.001,20,1024)

    sincos = sin(kd/2.)*cos(kd/2.)
    bipole_norm = sincos+(kd/2.)
    dipole_norm = (kd / 2.) - sincos

    plt.figure()
    plt.plot(kd,bipole_norm,label=r"$\scrP$$_{avg}$, Bipole")
    plt.plot(kd,dipole_norm,label=r"$\scrP$$_{avg}$, Dipole")
    plt.axhline(1,color="black",linestyle="--",label=r"$\scrP$$_{avg}$, Monopole")
    plt.axvline(1,color="black",linestyle=":")
    plt.xlim(0,20)
    plt.xlabel("kd")
    plt.ylabel("Normalized Power")
    plt.grid(which="both")
    plt.legend()
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))