# PSUACS
# ACS515_HW3
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/2/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    N=1024
    kr = np.linspace(0.1,10.,N)
    i = -1j
    u_r = (1./(kr**2))+(1j*(1./kr))

    for i in range(N):
        if np.real(u_r[i]) >= 0.99*np.abs(u_r[i]):
            if np.real(u_r[i+1]) < 0.99*np.abs(u_r[i+1]):
                print "real: " + str(kr[i])
                near = kr[i]

        elif np.imag(u_r[i]) >= 0.99*np.abs(u_r[i]):
            print "imag: " + str(kr[i])
            far = kr[i]
            break


    plt.figure()
    plt.semilogx(kr,np.abs(u_r))
    plt.semilogx(kr,np.real(u_r))
    plt.semilogx(kr,np.imag(u_r))
    plt.axvline(near)
    plt.axvline(far)
    plt.xlabel("kr")
    plt.ylabel("Particle Velocity Amplitude [m/s]")
    plt.xlim(kr[0],kr[-1])
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))