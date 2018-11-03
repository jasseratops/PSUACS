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
    #part3(data_3BB, fs, show=True)
    dataFiltered = part3(data_3BB,fs,show=False)
    part4(dataFiltered,fs)
    return 0

def part1(data,fs,ov,win,recLength):
    t = 11.
    N11 = int(t*fs)
    frst11 = data[0:N11]
    scnd11 = data[N11:N11*2]

    Gxx_1,freq,delF,Gxx_1_array = sigA.spectroArray(frst11,fs,recLength,sync=0,overlap=ov,winType=win)
    Gxx_2,_,_,_ = sigA.spectroArray(scnd11,fs,recLength,sync=0,overlap=ov,winType=win)

    m = np.shape(Gxx_1_array)[0]
    print m

    sigA.rms(frst11*sigA.window(win,len(frst11)))
    rms_Gxx_Avg = sum(Gxx_1)*delF
    print "rms_Gxx_Avg: " + str(rms_Gxx_Avg)

    plt.figure()
    plt.semilogy(freq,Gxx_1,label="0-11s")
    plt.semilogy(freq,Gxx_2,label="11-12s")
    plt.xlim(0,10E3)
    plt.ylim(1E-10,1E-3)
    plt.title(r"$G_{xx}$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Power Spectral Density [${Pa^2}$/Hz]")
    plt.legend()
    plt.savefig("Part 1.png")
    plt.show()

    return 0

def part2(data1,data2,fs,ov,win,recLength):
    t = 11.
    N11 = int(t*fs)
    frst11_1bb = data1[:N11]
    scnd11_1bb = data1[N11:2*N11]

    frst11_3bb = data2[:N11]
    scnd11_3bb = data2[N11:2*N11]

    frstCoh, freq = sigA.coherence(x_time= frst11_1bb,y_time=frst11_3bb,fs = fs,sliceLength=recLength,sync=0,overlap=ov,winType=win)
    scndCoh, _  = sigA.coherence(x_time=scnd11_1bb,y_time=scnd11_3bb,fs = fs,sliceLength=recLength,sync=0,overlap=ov,winType=win)

    plt.figure()
    plt.plot(freq,abs(frstCoh),label="0-11")
    plt.plot(freq,abs(scndCoh),label="11-22")
    plt.xlim(0,10.E3)
    plt.title("Coherence")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$\gamma^2$")
    plt.legend()
    plt.savefig("Part 2.png")
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
    theta = np.degrees(np.angle(h))
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
        plt.axvline(f1,color="black")
        plt.axvline(f2,color="black")
        plt.title("Butterworth BP [1kHz-6kHz], Magnitude")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.subplot(212)
        plt.semilogx(freqs,theta)
        plt.axvline(f1,color="black")
        plt.axvline(f2,color="black")
        plt.title("Butterworth BP [1kHz-6kHz], Phase")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase [Degrees]")
        plt.savefig("Part3, Butterworth BP.png")
        plt.subplots_adjust(hspace=0.55)

        plt.figure()
        plt.plot(times,tot)
        plt.xlim(3.630,3.640)
        plt.title(r"sin(2$\pi$t(100Hz)) + sin(2$\pi$t(2kHz)) + sin(2$\pi$t(10kHz))")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [WU]")
        plt.savefig("Part3, fuzzy sine.png")

        plt.figure()
        plt.plot(times,totFilt, label= "Filtered Sine")
        plt.plot(times,sin2k, label= "sin(2$\pi$t(2kHz))")
        plt.xlim(3.630,3.640)
        plt.title("Filtered sine")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [WU]")
        plt.legend()
        plt.savefig("Part3, filtered sine.png")


        plt.figure()
        plt.plot(times,data1,label = "Raw")
        plt.plot(times,data1Filt, label = "Filtered")
        plt.title("N96_2017_3bb, raw vs filtered")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [Pa]")
        plt.legend()
        plt.savefig("Part 3, raw vs filtered.png")
        plt.show()

    return data1Filt


def part4(data,fs):
    N = len(data)
    times = sigA.timeVec(N,fs)[1*fs:12*fs]
    one_twelve_3bb = data[1*fs:12*fs]

    recLength = int(1.*fs)
    Nrecs = 10

    slices = np.zeros((Nrecs,int(fs*1.)))

    for i in range(Nrecs):
        slices[i,]=one_twelve_3bb[i*recLength:(i+1)*recLength]
        if i>0:
            R_XY,tau = sigA.crossCorr(slices[0,],slices[i,],fs)
            td = tau[np.argmax(R_XY)]
            samd = int(td * fs)
            slices[i,] = np.roll(slices[i],samd)

    plt.figure()
    for i in range(Nrecs):
        plt.plot(times[:int(fs*1.)],slices[i,],label=str(i))
        plt.legend()

    slice_avg = np.mean(slices,axis=0)
    x_n_avg = sigA.timeAvg(one_twelve_3bb,fs,recLength,Nrecs)[0]

    plt.figure()
    plt.plot(times[:int(fs*1.)],slices[0,],label="0-1")
    plt.plot(times[:int(fs*1.)],x_n_avg,label="Avg")
    plt.title("Single Record vs Time-Sync Averaging")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [Pa]")
    plt.xlim(times[0],times[fs])
    plt.legend()
    plt.savefig("Part4.png")


    plt.figure()
    plt.plot(times[:int(fs*1.)],slices[0,],label="0-1")
    plt.plot(times[:int(fs*1.)],x_n_avg,label="avg")
    plt.plot(times[:int(fs*1.)],slice_avg,label="Delay Corr")
    plt.title("Single vs Time-Sync Avg'ing vs Time-Sync Avg'ing w/ Delay Correction")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [Pa]")
    plt.xlim(times[0],times[fs])
    plt.legend()
    plt.savefig("Part4, Improved.png")

    plt.show()

if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))