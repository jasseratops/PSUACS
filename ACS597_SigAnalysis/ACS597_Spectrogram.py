import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy import sin, pi, shape
import sigA
import sounddevice as sd

#foldername = "/Users/macbookpro/PycharmProjects/PSUACS/ACS597_SigAnalysis/"
foldername = "C:/Users/alshehrj/PycharmProjects/PSUACS/ACS597_SigAnalysis/"

def spectroArray(x_time, fs, sliceLength, sync=0,overlap=0,winType="uniform"):
    Gxx = np.zeros((1,sliceLength/2))

    i = 0
    while True:
        n = i * (int(sliceLength*(1-overlap))+sync)
        sliceEnd = n + sliceLength - 1

        if sliceEnd >= len(x_time)-1:
            break

        if i == 0:
            Gxx[0,]= sigA.ssSpec(x_time[n:sliceEnd], fs,winType)
        else:
            Gxx = np.concatenate((Gxx,[sigA.ssSpec(x_time[n:sliceEnd], fs,winType)]),axis=0)
        i += 1

    #### Gxx Avg
    GxxAvg = np.mean(Gxx, axis=0)
    freqAvg = sigA.freqVec(sliceLength, fs)[:int(sliceLength/2)]
    _, delF_Avg, _ = sigA.param(sliceLength, fs, show=False)

    print "Stepping out of Gxx"
    print np.shape(Gxx)
    return GxxAvg, freqAvg, delF_Avg, Gxx

def spectrogram(x_time, fs, sliceLength, sync=0, overlap=0, dB=True, winType="uniform", scale=True):
    N = len(x_time)
    Nslices = int(N / sliceLength)
    T = Nslices * sliceLength / fs

    _, freqAvg, _, Gxx = spectroArray(x_time=x_time, fs=fs, sliceLength=sliceLength, sync=sync, overlap=overlap, winType=winType)

    GxxRef = 1.0                            # V^2/Hz
    Gxx_dB = 10 * np.log10(Gxx / GxxRef)

    ext = [0, T, 0, fs / 2]
    color = "jet"
    if dB:
        plt.imshow(Gxx_dB.T, aspect="auto", origin="lower", cmap=color, extent=ext)
    else:
        plt.imshow(Gxx.T, aspect="auto", origin="lower",cmap=color, extent=ext)
    if scale:
        plt.ylim(ext[1] + 1, ext[3] * 0.8)

def main(args):
    #testing()
    actual()
    #recording()

def testing():
    fs = 2048.0
    T = 6.0
    N=int(fs*T)
    print N

    times = sigA.timeVec(N,fs)
    delT,delF,_= sigA.param(N,fs)

    f = 128

    x_time = np.zeros(N)

    for i in range(N):
        if i*delT < 2.0:
            x_time[i] += 0
        elif i*delT < 4.0:
            x_time[i] += sin(2*pi*f*times[i])
        else:
            x_time[i] += 0




    sliceLength = 256           # Length of single record
    ov = 0.0                    # Overlap

    spectrogram(x_time,fs,sliceLength,sync=0,dB=True,overlap=ov,winType="uniform")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()


def actual():
    filename = "T4_C5_3L_dec4a.wav"
    path = foldername+filename
    fs , data = wavfile.read(path)

    Nwav = len(data)
    print data.dtype
    print data
    if data.dtype != np.float32:
        print "Converting from " + str(data.dtype) + " to float32"
        data = data.astype(np.float32)
        data = data /32768.0

    print fs
    print data
    print Nwav
    print Nwav/float(fs)
    print 10 * "-"

    #t = sigA.timeVec(Nwav, fs)

    ov = 0.75
    sliceLength = 1024

    plt.figure()
    spectrogram(data,fs,sliceLength,sync=0,dB=True,overlap=ov,winType="hann")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()


def recording():
    fs = 48000
    T = 10
    N = fs*T

    x_time = sd.rec(frames=N,samplerate=fs,channels=1,blocking=True)

    ov = 0.75
    sliceLength = 1024

    plt.figure()
    spectrogram(x_time, fs, sliceLength, sync=0, dB=True, overlap=ov, winType="hann")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))

