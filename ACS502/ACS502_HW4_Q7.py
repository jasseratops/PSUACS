# PSUACS
# ACS502_HW4_Q7
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/13/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    f = np.linspace(20.,20.E3,1000)
    omega = 2*pi*f
    c = 343.
    rho0= 1.21
    k = omega/c
    l0 = 1.5875E-3
    a = 1.5875E-3
    r0 = 0.0254
    beta = 0.785
    S_pipe = pi*(r0**2)
    S_hole = pi*(a**2)
    Z0 = rho0*c

    leff = l0 + beta*a

    Z_pipe = Z0/S_pipe
    Z_hole = (Z0/pi)+1j*(omega*rho0*leff/S_hole)

    T = (Z_hole/(Z_hole+(0.5*Z_pipe)))
    Tw = (np.abs(T))**2
    TL = -10*np.log10(Tw)

    T2 = ((k**2)+(1j*k*leff/(a**2)))/((k**2)+(1j*k*leff/(a**2))+(1/((2)*(r0**2))))
    Tw2= (np.abs(T2))**2
    TL2 = -10*np.log10(Tw2)


    for i in range(len(f)):
        if TL[i] <= 3.:
            f3dB = f[i]
            break

    T3 = (f/f3dB)/(np.sqrt(1.+((f/f3dB)**2)))
    Tw3 = (np.abs(T3))**2
    TL3 = -10*np.log10(Tw3)

    plt.figure()
    plt.semilogx(f,TL)
    plt.semilogx(f,TL2)
    plt.semilogx(f,TL3)
    plt.axvline(f3dB)
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))