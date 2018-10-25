import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy import sin, pi, shape
import sigA
import sounddevice as sd

#foldername = "/Users/macbookpro/PycharmProjects/PSUACS/ACS597_SigAnalysis/"
foldername = "C:/Users/alshehrj/PycharmProjects/PSUACS/ACS597_SigAnalysis/"


def spectrogram(x_time, fs, sliceLength, sync=0, overlap=0,color="jet", dB=True, winType="uniform", scale=True):
    N = len(x_time)
    Nslices = int(N / sliceLength)
    T = Nslices * sliceLength / float(fs)
    print "T: " + str(T)

    _, freqAvg, _, Gxx = sigA.spectroArray(x_time=x_time, fs=fs, sliceLength=sliceLength, sync=sync, overlap=overlap, winType=winType)

    GxxRef = 1.0                            # V^2/Hz
    Gxx_dB = 10 * np.log10(Gxx / GxxRef)

    ext = [0, T, 0, fs / 2]

    if dB:
        plt.imshow(Gxx_dB.T, aspect="auto", origin="lower", cmap=color, extent=ext)
    else:
        plt.imshow(Gxx.T, aspect="auto", origin="lower",cmap=color, extent=ext)
    if scale:
        plt.ylim(ext[1] + 1, ext[3] * 0.8)


def main(args):
    sinTest()
    raceCar()
    #recording()


def sinTest():
    fs = 2048.0
    T = 6.0
    N=int(fs*T)
    print N

    times = sigA.timeVec(N,fs)
    delT,delF,_= sigA.param(N,fs)

    f = 128

    x_time = np.zeros(N)
    t = sigA.timeVec(N,fs)

    for i in range(N):
        if i*delT < 2.0:
            x_time[i] += 0
        elif i*delT < 4.0:
            x_time[i] += sin(2*pi*f*times[i])
        else:
            x_time[i] += 0

    plt.figure()
    plt.plot(t,x_time)

    plt.figure()

    sliceLength = 256           # Length of single record
    ov = 0                      # Overlap

    spectrogram(x_time,fs,sliceLength,sync=0,dB=False,color="YlOrRd",overlap=ov,winType="uniform",scale=False)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram, 128Hz Sine Wave")
    plt.show()


def raceCar():
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
    print np.shape(data)

    ov = 0.75
    sliceLength = 1024

    plt.figure()
    spectrogram(data,fs,sliceLength,sync=0,dB=True,overlap=ov,winType="hann")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Racecar Doppler Shift")
    plt.show()


def recording():
    fs = 44100
    T = 5
    N = fs*T

    print N

    recArray = sd.rec(frames=N,samplerate=fs,channels=1,blocking=True)

    x_time = np.reshape(recArray, (len(recArray),))

    t = sigA.timeVec(N,fs)

    plt.figure()
    plt.plot(t,x_time)

    ov = 0.75
    sliceLength = 2056

    GxxAvg = sigA.ssSpec(x_time=x_time,fs=fs)
    FreqAvg = sigA.freqVec(N,fs)

    plt.figure()
    plt.plot(FreqAvg[:len(GxxAvg)],GxxAvg)

    scaled = np.int16(x_time/np.max(np.abs(x_time)) * 32767)
    wavfile.write('test.wav', 44100, scaled)

    print np.shape(x_time)

    plt.figure()
    spectrogram(x_time=x_time, fs=fs, sliceLength=sliceLength, sync=0, dB=True, overlap=ov, winType="hann",scale=True)
    #spectroArray(x_time, fs, sliceLength, sync=0, overlap=ov, winType="hann")

    plt.title("Electric Guitar, Cmajor Chord + tremolo")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))