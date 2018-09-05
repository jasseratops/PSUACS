# PSUACS
# ACS597_SigA_Analyzer
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/3/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import sounddevice as sd
import sigA


def main(args):
    fs = 44100.00
    T = 0.1
    N = int(fs*T)
    fakeFrq = 4500.0
    gap = np.zeros(1024)


    t= sigA.timeVec(N=N,fs=fs)
    freq = sigA.freqVec(N=N, fs=fs)

    fake =np.zeros(N)

    for i in range(len(fake)):
        fake[i] = 0.01*sin(2*pi*fakeFrq*t[i])

    '''
    print "Starting Recording"
    recArray = sd.rec(frames=N,samplerate=fs,channels=1,blocking=True)
    print "Stopped recording"
    '''
    #recArray = np.append(gap,fake[len(gap):])
    #print np.shape(recArray)

    recArray = fake

    print "Playing Recording"
    sd.play(recArray,samplerate=fs,blocking=True)
    print "Stopped playing"

    plt.figure()
    plt.plot(t,recArray)
    plt.show()

    Gxx = sigA.ssSpec(recArray,fs)

    plt.figure()
    plt.semilogy(freq[0:len(Gxx)],Gxx)
    plt.show()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
