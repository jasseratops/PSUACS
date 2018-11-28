# PSUACS
# DopplerShiftLFM
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/21/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import scipy.signal as sig
import sigA

def main(args):
    fs = 48000
    T = 1.
    N = int(fs*T)
    times = sigA.timeVec(N,fs)
    fsrc = np.zeros(N)
    frcvr = np.zeros(N)
    V = 100.
    c = 343.
    meth = "linear"
    dur = 0.02
    durInd = int(fs*dur)

    f0src = 100.
    f1src = 4000.
    f0rcvr = f0src * (1 - (V / c)) / (1 + (V / c))
    f1rcvr = f1src * (1 - (V / c)) / (1 + (V / c))

    print(f0rcvr)
    print(f1rcvr)

    durDop = dur*((1+(V/c))/(1-(V/c)))

    print durDop
    durDopInd = int(durDop*fs)

    LFM = sig.chirp(times[:durInd],f0=f0src,f1=f1src,t1=dur,method=meth)
    print len(LFM)
    fsrc[:durInd]= LFM

    LFM_Dop = sig.chirp(times[:durDopInd],f0=f0rcvr,f1=f1rcvr,t1=durDop,method=meth)
    print len(LFM_Dop)
    frcvr[:durDopInd] = LFM_Dop

    R_XY, tau = sigA.crossCorr(frcvr,fsrc,fs)

    plt.figure()
    plt.plot(times,fsrc)
    plt.plot(times,frcvr)

    plt.figure()
    plt.plot(tau,R_XY)
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))