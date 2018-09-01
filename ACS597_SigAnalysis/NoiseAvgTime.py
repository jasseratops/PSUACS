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


    for i in range(runs):
        x = generator(baseline)
        sum = sum + x

    sum = sum/runs


    plt.figure()
    plt.plot(x)
    plt.plot(sum)
    plt.plot(baseline)
    plt.show()

    return 0


def generator(baseline):
    N = len(baseline)
    x_time = np.zeros(N)

    for i in range(N):
        x_time[i] = baseline[i]+(rn.random()-0.5)

    return x_time


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))