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
import scipy.signal as sig
import sounddevice as sd


def main(args):
    data_1BB, fs = sf.read("N96_2017_1BB.wav")
    data_3BB, _ = sf.read("N96_2017_3BB.wav")
    win ="hann"
    ov = 0.5

    recDuration = 0.1
    recLength = recDuration*fs

    #part1(data_3BB, fs,ov,win,recLength)
    #part2(data_1BB,data_3BB,fs,ov,win,recLength)
    dataFiltered = part3(data_3BB,fs,show=False)
    part4(dataFiltered,fs)
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
    Gxx_2,_,_,_ = sigA.spectroArray(scnd11,fs,recLength,sync=0,overlap=ov,winType=win)

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
    plt.semilogy(freq,Gxx_2)
    plt.xlim(0,10E3)
    plt.ylim(1E-10,1E-3)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [Pa^2/Hz]")

    plt.show()

    return 0

def part2(data1,data2,fs,ov,win,recLength):
    t = 11.
    N11 = int(t*fs)
    frst11_1bb = data1[:N11]
    scnd11_1bb = data1[N11:2*N11]

    frst11_3bb = data2[:N11]
    scnd11_3bb = data2[N11:2*N11]

    '''
    S_XX =       sigA.dsSpec(frst11_1bb,fs,win)
    S_XX_cross = sigA.crossSpec(frst11_1bb,frst11_1bb,fs,win)

    print len(S_XX)
    print len(S_XX_cross)

    for i in range(len(S_XX)):
        if S_XX[i]-S_XX_cross[i] > (1.E-19):
            print i
            print S_XX[i]
            print S_XX_cross[i]
            print S_XX[i]-S_XX_cross[i]
            print 10*"-"
            
    plt.figure()
    plt.plot(S_XX-S_XX_cross)
    plt.figure()
    plt.plot(S_XX)
    #plt.plot(S_XX_cross)
    
    G_XX =       sigA.ssSpec(frst11_1bb,fs,win)
    G_XX_cross = sigA.ssCrossSpec(frst11_1bb,frst11_1bb,fs,win)

    plt.figure()
    plt.plot(G_XX)
    plt.plot(G_XX_cross)

    '''
    frstCoh, freq = sigA.coherence(x_time= frst11_1bb,y_time=frst11_3bb,fs = fs,sliceLength=recLength,sync=0,overlap=ov,winType=win)
    scndCoh, _  = sigA.coherence(x_time=scnd11_1bb,y_time=scnd11_3bb,fs = fs,sliceLength=recLength,sync=0,overlap=ov,winType=win)

    plt.figure()
    plt.plot(freq,abs(frstCoh),label="0-11")
    plt.plot(freq,abs(scndCoh),label="11-22")
    plt.xlim(0,10.E3)
    plt.title("Coherence")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$\gamma^2$")
    plt.show()

    return 0

def part3(data1,fs,show):
    N = len(data1)
    times = sigA.timeVec(N,fs)

    Nyq = fs/2.
    f1 = 1.E3
    f2 = 6.E3
    Wn = [f1/Nyq,f2/Nyq]

    b,a = sig.butter(10,Wn,btype ="bandpass")

    w,h = sig.freqz(b,a)
    theta = np.angle(h)
    mag = 20*np.log10(abs(h))
    freqs = w*Nyq/(pi)

    ### Sine Test
    f100 = 100.
    f2k = 2000.
    f10k = 1.E4

    sin100 = sin(2*pi*f100*times)
    sin2k = sin(2*pi*f2k*times)
    sin10k = sin(2*pi*f10k*times)
    tot = sin2k+sin100+sin10k
    totFilt = sig.lfilter(b,a,tot)
    ###

    data1Filt = sig.lfilter(b,a,data1)
    if show:
        plt.figure()
        plt.subplot(211)
        plt.semilogx(freqs,mag)
        plt.subplot(212)
        plt.semilogx(freqs,theta)
        plt.axvline(f1)
        plt.axvline(f2)

        plt.figure()
        plt.plot(times,tot)

        plt.figure()
        plt.plot(times,totFilt)
        plt.plot(times,sin2k)
        plt.xlim(3.638,3.640)

        plt.figure()
        plt.plot(times,data1)
        plt.plot(times,data1Filt)
        plt.title("N96_2017_3bb, raw vs filtered")
        plt.show()
    return data1Filt

def part4(data,fs):
    N = len(data)
    times = sigA.timeVec(N,fs)
    one_eleven = data[1*fs:11*fs]

    recLength = int(0.25*fs)
    period = int(1.*fs)
    sync = abs(period-recLength)

    data_avg,timeAvg,_,x_n = sigA.timeAvg(one_eleven,fs,period,Nrecs=10,sync=0)

    for i in range(10):
        print "data_avg"
        sd.play(data_avg,fs,blocking=True)

    for i in range(10):
        print "x_n"
        sd.play(x_n[i,], fs, blocking=True)

    plt.figure()
    plt.plot(times,data)
    plt.xlim(1,11)

    plt.figure()
    plt.plot(timeAvg,x_n[2,])
    plt.plot(timeAvg,data_avg)
    plt.show()

if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))