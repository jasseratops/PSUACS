# PSUACS
# ACS502_HW5_Q7
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/30/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    kr = np.linspace (0.,30.,1024)
    z = ((kr**2)/(1+(kr**2)))+1j*((kr)/(1+(kr**2)))

    z_re = z.real
    z_im = z.imag

    plt.figure()
    plt.plot(kr,z_re, label="Resistance")
    plt.plot(kr,z_im,label = "Reactance")
    plt.title("Specific Acoustic Impedance of Sphere")
    plt.xlabel("kr")
    plt.ylabel("Impedance [rho0*c]")
    plt.xlim(kr[0],kr[-1])
    plt.ylim(0,1.1)
    plt.legend()
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))