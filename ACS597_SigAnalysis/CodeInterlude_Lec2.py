# PSUACS
# CodeInterlude
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/21/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    f0 = 200
    fs = 420
    T = 0.5
    A = 3.5

    dt = 1.0/fs
    N=round(T/dt)

    times = np.arange(0,N)*dt
    sine_wave = A*sin(2*pi*f0*times)

    plt.figure()
    plt.plot(times,sine_wave)
    plt.title("Sine Wave")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.xlim(min(times),max(times))
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))