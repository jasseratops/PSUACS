import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt
import sigA

def main(args):
    fs = 48000.
    T = 2.
    N = int(T*fs)
    print N
    times = sigA.timeVec(N,fs)
    f = 100.

    stim = sin(2*pi*f*times[:N/2])
    x = np.zeros(N)
    startInd = int(0*fs)
    endInd = int(startInd+len(stim))
    x[startInd:endInd] = stim
    #x = np.append(stim,stim)

    sliceNum = 1024
    plt.figure()
    plt.plot(times,x)

    plt.figure()
    for i in range(18):
        xSlice = x[i*sliceNum:(i+1)*sliceNum]
        auto,autoTau = sigA.autocor(xSlice,fs)
        cross,crossTau = sigA.crossCorr(xSlice,xSlice,fs)
    plt.plot(autoTau, np.abs(auto))
    plt.plot(crossTau, np.abs(cross))


    plt.figure()
    sigA.crossCorrSpectrogram(x,x,fs,sliceNum,pad=True)

    xSlice = x[int(0.99*fs):][:sliceNum]
    plt.figure()
    auto,autoTau = sigA.autocor(xSlice,fs)
    cross,crossTau = sigA.crossCorr(xSlice,xSlice,fs)
    plt.plot(autoTau, np.abs(auto))
    plt.plot(crossTau, np.abs(cross))



    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))