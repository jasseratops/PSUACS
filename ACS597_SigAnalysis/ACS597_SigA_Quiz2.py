# PSUACS
# ACS597_SigA_Quiz2
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/24/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import soundfile as sf
import sigA


def main(args):
    part1()
    return 0

def part1():
    data, fs = sf.read("N96_2017_3BB.wav")
    N = len(data)
    times = sigA.timeVec(N,fs)
    t = 11.
    frst11 = data[0:int(fs*t)]
    scnd11 = data[int(fs*t):]

    #msAvgd = sigA.

    recDuration = 0.1
    recLength = recDuration*fs
    win = "hann"
    ov = 0.5

    Gxx_1,freq,delF,Gxx_1_array = sigA.spectroArray(frst11,fs,recLength,sync=0,overlap=ov,winType=win)
    Gxx_2,_,_,_ = sigA.spectroArray(scnd11,fs,recLength,sync=0,overlap=ov,winType=win)

    m = np.shape(Gxx_1_array)[0]
    print m
    Gxx_tot = sigA.ssSpec(frst11[0:int(m*recLength)],fs,winType=win)
    _,delF_tot,_ = sigA.param(N,fs,show=False)


    sigA.rms(frst11)
    print sum(Gxx_1)*delF
    print sum(Gxx_tot)*delF_tot

    plt.figure()
    plt.semilogy(freq,Gxx_1)
    plt.semilogy(freq,Gxx_2)
    plt.xlim(0,10E3)
    plt.ylim(1E-10,1E-3)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [Pa^2/Hz]")


    plt.figure()
    plt.plot(times,data)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    plt.title("")
    #plt.show()

    return 0
if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))