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
    data_1BB, fs = sf.read("N96_2017_1BB.wav")
    data_3BB, _ = sf.read("N96_2017_3BB.wav")

    win ="hann"
    ov = 0.5

    recDuration = 0.1
    recLength = recDuration*fs

    #part1(data_3BB, fs,ov,win,recLength)
    part2(data_1BB,data_3BB,fs,ov,win,recLength)
    return 0

def part1(data,fs,ov,win,recLength):
    N = len(data)
    times = sigA.timeVec(N,fs)
    t = 11.
    N11 = int(t*fs)
    frst11 = data[0:N11]
    scnd11 = data[N11:N11*2]

    #msAvgd = sigA.




    Gxx_1,freq,delF,Gxx_1_array = sigA.spectroArray(frst11,fs,recLength,sync=0,overlap=ov,winType=win)
    #Gxx_2,_,_,_ = sigA.spectroArray(scnd11,fs,recLength,sync=0,overlap=ov,winType=win)

    m = np.shape(Gxx_1_array)[0]
    print m

    sigA.rms(frst11*sigA.window(win,len(frst11)))
    rms_Gxx_Avg = sum(Gxx_1)*delF
    print "rms_Gxx_Avg: " + str(rms_Gxx_Avg)

    ''' Gxx_tot Tangent
    Nlong = int(m*recLength)
    Gxx_tot = sigA.ssSpec(frst11[:Nlong],fs,winType=win)
    _,delF_tot,_ = sigA.param(Nlong,fs,show=False)
    rms_Gxx_tot = sum(Gxx_tot)*delF_tot
    print "rms_Gxx_tot: " + str(rms_Gxx_tot)
    '''

    plt.figure()
    plt.semilogy(freq,Gxx_1)
    #plt.semilogy(freq,Gxx_2)
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

def part2(data1,data2,fs,ov,win,recLength):
    t = 11.
    N11 = int(t*fs)
    frst11_1bb = data1[:N11]
    scnd11_1bb = data1[N11:2*N11]

    frst11_3bb = data2[:N11]
    scnd11_3bb = data2[N11:2*N11]

    frstCoh, freq = sigA.coherence(frst11_1bb,frst11_3bb,fs,recLength,sync=0,overlap=ov,winType=win)
    scndCoh, _  = sigA.coherence(scnd11_1bb,scnd11_3bb,fs,recLength,sync=0,overlap=ov,winType=win)

    plt.figure()
    plt.plot(freq,abs(frstCoh))
    plt.plot(freq,abs(scndCoh))
    plt.xlim(0,10.E3)
    plt.show()

    return 0
if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))