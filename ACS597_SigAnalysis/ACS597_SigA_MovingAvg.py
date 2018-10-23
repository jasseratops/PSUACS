# PSUACS
# ACS597_SigA_MovingAvg
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/17/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import scipy.signal as sig
import sigA

def main(args):
    fs = 1000.
    T = 20.
    N = int(fs*T)

    delT,_,_ = sigA.param(N,fs)

    Nc = 2
    b = np.ones(Nc,dtype=float)*delT
    impTrain = np.zeros(128)
    impTrain[0] = 1./delT
    impResp = sig.lfilter(b,1.,impTrain)

    bdiff = np.array([1,-1])*delT
    impRespDiff = sig.lfilter(bdiff,1.,impTrain)

    w,h = sig.freqz(b,1)

    print impResp
    plt.figure()
    plt.plot(impResp)

    plt.figure()
    plt.plot(impRespDiff)

    plt.figure()
    plt.plot(w,20*np.log10(np.abs(h)))
    plt.show()



    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))