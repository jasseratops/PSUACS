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

    mic2 = sf.read(file=files[0],frames=durInd,start=startTimeInd)[0]
    mic1 = sf.read(file=files[1],frames=durInd,start=startTimeInd)[0]


    mic2pad = np.zeros(len(mic2)*2)
    mic1pad = np.zeros(len(mic1)*2)

    for i in range(len(mic2)):
        mic2pad[i * 2] = mic2[i]
        mic1pad[i * 2] = mic1[i]

    mic2pad2 = np.append(mic2,np.zeros(len(mic2)))
    mic1pad2 = np.append(mic2, np.zeros(len(mic1)))

    times = sigA.timeVec(N,fs)

    micDist13 = 1.5
    c = 343.

    sliceLength = 1024*2


    maxDel = micDist13/c

    fc = 100.

    b,a = sig.butter(N=4,Wn=fc/(fs/2),btype="high")
    w,h = sig.freqz(b,a)
    mag = 20*np.log10(abs(h))

    freqs = (w/pi)*(fs/2)
    fTest = 2000.
    testSig = sin(2*pi*fTest*times)
    filtTestSig = sig.lfilter(b,a,testSig)

    mic2Filt = sig.lfilter(b,a,mic2)
    mic1Filt = sig.lfilter(b,a,mic1)


    ### LECTURE 11
    plt.figure()
    #sigA.crossCorrSpectrogram(mic2,mic1,fs,sliceLength)
    #sigA.crossCorrSpectrogram(mic2pad,mic1pad,fs*2,sliceLength)
    sigA.crossCorrSpectrogram(mic2pad2, mic1pad2, fs, sliceLength)
    plt.axhline(maxDel)
    plt.axhline(-maxDel)
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
    sigA.crossCorrSpectrogram(mic2Filt,mic1Filt,fs,sliceLength)
    plt.draw()
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))