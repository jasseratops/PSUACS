# JAscripts
# ampMod
# Jasser Alshehri
# Starkey Hearing Technologies
# 3/17/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    fs = 44.1E3
    startTime = 0
    endTime = 1
    t = np.linspace(startTime,endTime,(endTime-startTime)*fs)

    omegaM = 5
    omegaC = 100

    xm = sin(2*pi*omegaM*t)
    xc = sin(2*pi*omegaC*t)

    mod1 = xm*xc
    mod2 = (1+xm)*xc

    mod1FFT = np.fft.fft(mod1)
    mod2FFT = np.fft.fft(mod2)
    freq = np.fft.fftfreq(t.shape[-1])

    plt.figure()
    plt.plot(t,xm)
    plt.plot(t,xc)

    plt.figure()
    plt.plot(t,mod1)
    plt.plot(t,xm)

    plt.figure()
    plt.plot(t,mod2)
    plt.plot(t,xm+1)

    plt.figure()
    plt.plot(mod1FFT.real)

    plt.figure()
    plt.plot(mod2FFT.real)

    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))