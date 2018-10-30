# PSUACS
# ACS502_HW5_Q2_SawCastor
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/27/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import scipy.signal as sig


def main(args):
    t = np.linspace(0,1,500)
    saw = sig.sawtooth(2*pi*4*t)

    N = len(t)
    T = max(t)
    fs = N/T


    delF = 1./T
    freq = np.arange(0,N)*delF
    fft2 = np.zeros(len(freq))

    sawFFT = abs(np.fft.fft(saw))


    n = np.arange(1.,400*1025.)
    coeff = 2./pi
    for i in range(len(freq)):
        for x in n:
            fft2[i] = coeff*sum((1./n)*sin(n*2*pi*(freq[i])))


    plt.figure()
    plt.plot(t,saw)

    plt.figure()
    plt.plot(freq[:N/2],20*np.log10(sawFFT)[:N/2])

    plt.figure()
    plt.plot(freq[:N/2],fft2[:N/2])
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))