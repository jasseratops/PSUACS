# PSUACS
# ACS519_HW1
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/23/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    k = 1.
    r = np.linspace(1,50,1024*2)
    x = exp(1j*k*r)/(4*pi*r)

    plt.figure()
    plt.plot(r,x.real)
    plt.plot(r,x.imag)

    plt.xlim(0,r[-1])
    plt.show()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))