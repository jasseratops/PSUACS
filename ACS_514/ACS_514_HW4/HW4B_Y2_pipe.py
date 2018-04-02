import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi


c = 343                                                         # m/s
rho = 1.2                                                       # kg/m^3
rad = 25E-3                                                     # radius
A = pi*(rad**2)

freqs = np.linspace(100,2000,1901)

YmagMtx = np.zeros(len(freqs))                                  # initialize admittance matrix

def calcY2(L,N,EXACT):

    Ymtx = np.zeros(len(freqs))                                 # initialize admittance matrix

    for i in range(len(freqs)):
        w = freqs[i]*2*pi                                       # Calc angular frequency
        k = w/c                                                 # Calc wave number
        alpha = k*L                                             # Calc alpha
        Za = (1j*w*rho*L)/(2*A*N)                               # Define system mass
        Zb = (rho*(c**2)*N)/(1j*w*A*L)                          # Define system compliance

        if EXACT:                                               # Define exact solution model
            U2 = (-A * k * sin(alpha)) / (1j * w * rho)
            P2 = cos(alpha)
        else:                                                   # Define slice solution model
            Tmtx = np.matrix([[1+(Za/Zb), Za*(2+(Za/Zb))],
                              [1/Zb, 1+(Za/Zb)]])

            U2 = np.matrix([0, 1]) * np.linalg.matrix_power(Tmtx,N) * np.matrix([[1], [0]])
            P2 = np.matrix([1, 0]) * np.linalg.matrix_power(Tmtx,N) * np.matrix([[1], [0]])


        Y2 = U2/P2                                              # Define admittance according to U2 and P2

        YmagMtx[i] = abs(Y2)                                    # input magnitude of admittance into admittance matrix

    return YmagMtx


plt.subplot(211)                                                # plot solutions for N = 4
plt.semilogy(freqs, abs(calcY2(0.25,4,False)))
plt.semilogy(freqs, abs(calcY2(0.25,4,True)))
plt.xlim(100,2000)
plt.title('Admittance, N=4')
plt.grid(True)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Admittance (kg/s.m^4)")


plt.subplot(212)                                                # plot solutions for N = 20
plt.semilogy(freqs, abs(calcY2(0.25,20,False)))
plt.semilogy(freqs, abs(calcY2(0.25,20,True)))
plt.xlim(100,2000)
plt.title('Admittance, N=20')
plt.grid(True)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Admittance (kg/s.m^4)")

plt.show()