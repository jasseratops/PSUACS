# PSUACS
# NoiseAvg
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/31/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import random as rn


def main(args):
    N = 1024
    runs = 100
    sum = np.zeros(N)
    t = np.arange(N)
    baseline = sin(2*pi*4*t/N)
    bl_fft = np.fft.fft(baseline)



    plt.figure()
    plt.plot(x)
    plt.plot(sum)
    plt.plot(baseline)
    plt.show()


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))