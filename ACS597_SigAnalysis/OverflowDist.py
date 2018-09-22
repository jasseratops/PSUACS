# PSUACS
# OverflowDist
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/19/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import sigA

def main(args):
    fs = 12000
    T = 0.5
    N = T * fs
    times = sigA.timeVec(N,fs)
    f = 100.0

    x_time = sin(2*pi*f*times)

    threshFac = 0.8
    thresh = threshFac*max(x_time)

    for i in range(len(x_time)):
        if x_time[i] > thresh:
            x_time[i] -= thresh
        elif x_time[i] < -1*thresh:
            x_time[i] += thresh

    Gxx = sigA.ssSpec(x_time,fs)
    freqs = sigA.freqVec(N,fs)

    plt.figure()
    plt.plot(times,x_time)

    plt.figure()
    plt.plot(freqs[:len(Gxx)],20*np.log10(Gxx))
    plt.show()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))