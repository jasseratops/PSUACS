import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi


c = 343                             # m/s
rho = 1.2                           # kg/m^3
rad = 25E-3                         # radius
A = pi*(rad**2)

freqs = np.linspace(100,2000,1901)

YmagMtx = np.zeros(len(freqs))         # initialize admittance matrix

def calcY2(L,N,EXACT):

    Ymtx = np.zeros(len(freqs))  # initialize admittance matrix

    for i in range(len(freqs)):
        w = freqs[i]*2*pi
        k = w/c
        alpha = k*L

        if EXACT:
            Tmtx = np.matrix([[cos(alpha), (1j * w * rho * sin(alpha)) / (A * k)],
                             [(-A * k * sin(alpha)) / (1j * w * rho), cos(alpha)]])
        else:
            Tmtx = np.matrix([[1-(0.5*(alpha/N)), ((1j*alpha)/2*N)*(2-(0.5*((alpha/N)**2)))],
                             [(1j*alpha)/N, 1-(0.5*(alpha/N))]])


        U2 = np.matrix([0,1])*(Tmtx**N)*np.matrix([[1],[0]])
        P2 = np.matrix([1,0])*(Tmtx**N)*np.matrix([[1],[0]])

        Y2 = U2/P2

        YmagMtx[i] = abs(Y2)

    return YmagMtx


plt.subplot(211)
plt.semilogy(freqs, abs(calcY2(0.25, 4,False)))
plt.semilogy(freqs, abs(calcY2(0.25, 4,True)))
plt.xlim(100,2000)

plt.title('Admittance, Slice Solution.')
plt.grid(True)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Admittance")


plt.subplot(212)
plt.semilogy(freqs, abs(calcY2(0.25,20,False)))
plt.semilogy(freqs, abs(calcY2(0.25,20,True)))
plt.xlim(100,2000)
plt.title('Admittance, Exact Solution')
plt.grid(True)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Admittance")

plt.show()