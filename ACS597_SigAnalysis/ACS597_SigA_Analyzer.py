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
import random as rn
import scipy.signal as sig

def main(args):
    fs = 44100.00
    T = 2.0
    N = int(fs*T)

    t= sigA.timeVec(N=N,fs=fs)
    freq = sigA.freqVec(N=N, fs=fs)


    #####
    print "Starting Recording"
    recArray = sd.rec(frames=2*N,samplerate=fs,channels=1,blocking=True)[N:]
    print "Stopped recording"

    #####
    print "Playing Recording"
    sd.play(recArray,samplerate=fs,blocking=True)
    print "Stopped playing"


    x_time = np.reshape(recArray,(len(recArray),))

    Gxx = sigA.ssSpec(x_time,fs)


    plt.figure()
    plt.subplot(211)
    plt.plot(t,x_time)
    plt.title("Time Domain (whistling at 1kHz)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")

    plt.subplot(212)
    plt.semilogy(freq[0:len(Gxx)],Gxx)
    plt.title("Freq. Domain (whistling at 1kHz)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [WU^2 / Hz ???]")
    plt.show()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
