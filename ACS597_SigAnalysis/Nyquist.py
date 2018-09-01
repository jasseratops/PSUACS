# PSUACS
# Nyquist
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/30/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    N = 1024

    x_time = np.zeros(N)

    for i in range(N):
        x_time[i] = i%2

    print x_time

    X_fft = abs(np.fft.fft(x_time))

    plt.figure()
    plt.plot(X_fft)
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))