# PSUACS
# SubHarmonics
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/5/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import sigA

def main(args):
    fs = 44.1E3
    T = 0.5
    N = int(T*fs)

    f = 200.

    t = sigA.timeVec(N,fs)

    base = sin(2 * pi * (f) * t)

    ### Harmonics
    for i in range(50):
        if i > 1:
            base += (1./i)*sin(2*pi*(i*f)*t)

    for i in range(50):
        if i > 1:
            base += (1. / i) * sin(2 * pi * (f/i) * t)

    resp = np.fft.fft(base)


    plt.figure()
    plt.plot(t,base)

    plt.figure()
    plt.plot(abs(resp))
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))