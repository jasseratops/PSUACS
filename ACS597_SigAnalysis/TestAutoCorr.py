# PSUACS
# TestAutoCorr
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/19/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import sigA
import scipy.signal as sig

def main(args):
    fs = 756.E3

    T = 0.50
    N = int(T*fs)

    times = sigA.timeVec(N,fs)
    dur = 0.01

    f = 6000.

    delay = 0#17.5
    delay *= 1.E-6
    inDelay = int(delay*fs)
    waveA = np.zeros(N)
    waveA[:int(dur*fs)] = sig.chirp(times[0:int(dur*fs)],f0=2000,f1=1000,t1 = times[int(dur*fs)],method="linear")
    waveB = np.roll(waveA,inDelay)

    R_AY , tau = sigA.crossCorr(waveA,waveB,fs)

    timeShift = tau[np.argmax(R_AY)]
    print "timeShift: " + str(timeShift)


    plt.figure()
    plt.plot(times,waveA)
    plt.plot(times,waveB)

    plt.figure()
    plt.plot(tau,R_AY)
    plt.axvline(timeShift)

    plt.figure()
    plt.plot(times,waveA-waveB)
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))