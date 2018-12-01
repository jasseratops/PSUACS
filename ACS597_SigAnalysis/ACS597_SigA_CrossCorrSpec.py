# PSUACS
# ACS597_SigA_CrossCorrSpec
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/28/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import sigA
import sounddevice as sd
import soundfile as sf
import scipy.signal as sig

files = ["171013102503_1_C.wav","171013102503_2_C.wav","171013102503_3_C.wav","171013102503_4_C.wav"]

def main(args):
    _,fs = sf.read(file=files[0])

    startTime = 5.
    endTime = 13.
    duration = endTime - startTime

    startTimeInd = int(startTime*fs)
    durInd = int(duration*fs)
    N = durInd
    print durInd

    mic2 = sf.read(file=files[0],frames=durInd,start=startTimeInd)[0]
    mic1 = sf.read(file=files[2],frames=durInd,start=startTimeInd)[0]

    times = sigA.timeVec(N,fs)

    dist = 1.5
    c = 343.
    maxDel = dist/c

    sliceLength = 1024



    fc = 1000.

    b,a = sig.butter(N=4,Wn=fc/(fs/2),btype="high")
    w,h = sig.freqz(b,a)
    mag = 20*np.log10(abs(h))

    freqs = (w/pi)*(fs/2)
    fTest = 2000.
    testSig = sin(2*pi*fTest*times)
    filtTestSig = sig.lfilter(b,a,testSig)

    mic2Filt = sig.lfilter(b,a,mic2)
    mic1Filt = sig.lfilter(b,a,mic1)


    plt.figure()
    sigA.crossCorrSpectrogram(mic2,mic1,fs,sliceLength,pad=False)
    plt.axhline(maxDel,linestyle = ":",color="white")
    plt.axhline(-maxDel,linestyle = ":",color="white")
    plt.ylim(-0.008,0.008)
    plt.xlim(0,8)
    plt.title("CrossCorrelation Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Time Shift [s]")

    plt.figure()
    sigA.crossCorrSpectrogram(mic2,mic1,fs,sliceLength,pad=True)
    plt.axhline(maxDel,linestyle = ":",color="white")
    plt.axhline(-maxDel,linestyle = ":",color="white")
    plt.ylim(-0.008,0.008)
    plt.xlim(0,8)


    specSlice = 1024*2**4
    plt.figure()
    sigA.spectrogram(mic1,fs,specSlice,overlap=0.90,winType="hann")
    plt.ylim(0,4E3)
    '''
    plt.figure()
    plt.plot(freqs,mag)
    plt.axhline(max(mag)-3)

    plt.figure()
    plt.plot(times,testSig)
    plt.plot(times,filtTestSig)
    plt.xlim(1,1.01)
    '''
    plt.figure()
    TD = sigA.crossCorrSpectrogram(mic2Filt,mic1Filt,fs,sliceLength)
    plt.axhline(maxDel,linestyle = ":",color="white")
    plt.axhline(-maxDel,linestyle = ":",color="white")
    plt.ylim(-0.008,0.008)
    plt.xlim(0,8)
    plt.title("Cross Correlation Spectrogram, Filtered and Padded")
    plt.xlabel("Time [s]")
    plt.ylabel("Time Shift [s]")

    Nc = 20.

    bma = np.ones(int(Nc))/Nc
    ama = 1.
    TDtimes = np.linspace(0,8,len(TD))


    for i in range(len(TD)):
        if np.abs(TD[i]) > maxDel:
            TD[i] = (TD[i-1]+TD[i+1])/2
    TDavg = sig.lfilter(bma,ama,TD)

    h=17.
    V=60.
    CPA = 3.87
    CPA_ind = int(fs*CPA)


    tt = sigA.timeVec(len(mic1[CPA_ind:]),fs)
    tt_neg = sigA.timeVec(len(mic1[:CPA_ind]),fs)
    tt_neg = np.flip(tt_neg,0)*-1.
    tt = np.concatenate((tt_neg,tt))

    sinThet = V*tt/np.sqrt((h**2)+((V*tt)**2))
    cosThet = cos(np.arcsin(sinThet))

    a1 = h/cosThet
    d1 = h*(sinThet/cosThet)
    d2 = d1+dist
    a2 = np.sqrt((d2**2) +(h**2))
    TD_model = (a1-a2)/(c)

    plt.figure()
    sigA.crossCorrSpectrogram(mic2Filt,mic1Filt,fs,sliceLength)
    plt.plot(times,TD_model,label="Time Delay Model",color="red",linestyle="--")
    plt.title("Time Delay, Model vs Calc")
    plt.xlabel("Time [s]")
    plt.ylabel("Time Delay [s]")

    plt.xlim(0,8)
    plt.legend()

    plt.show()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))