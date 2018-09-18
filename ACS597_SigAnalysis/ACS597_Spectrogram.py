import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy import sin, pi, shape
import sigA

foldername = "/Users/macbookpro/PycharmProjects/PSUACS/ACS597_SigAnalysis/"

'''
def ssSpecAvg(x_time, fs, sliceLength, Nslices, sync=0,overlap=0,winType="uniform"):
    Gxx = np.zeros((Nslices, sliceLength / 2))
    for i in range(Nslices):
        n = i * (int(sliceLength*(1-overlap))+sync)
        Gxx[i] = sigA.ssSpec(x_time[n:n + sliceLength - 1], fs,winType)

    #### Gxx Avg
    GxxAvg = np.mean(Gxx, axis=0)
    freqAvg = sigA.freqVec(sliceLength, fs)[:int(sliceLength/2)]
    _, delF_Avg, _ = sigA.param(sliceLength, fs, show=False)
    print np.shape(Gxx)
    return GxxAvg, freqAvg, delF_Avg, Gxx
'''
def spectroArray(x_time, fs, sliceLength, Nslices, sync=0,overlap=0,winType="uniform"):
    Gxx = np.zeros((1,sliceLength/2))
    sliceEnd = 0
    i = 0
    n = 0
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

def main(args):
    testing()

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
        #x_time[i] = np.random.randn()
        if i*delT < 2.0:
            x_time[i] += 0
        elif i*delT < 4.0:
            x_time[i] += sin(2*pi*f*times[i])
        else:
            x_time[i] += 0


    sliceLength = 256
    Nslices = N/sliceLength
    sync = 0


    spectrogram(x_time,fs,sliceLength,sync=sync,overlap=0.0,winType="hann")

    plt.figure()
    plt.plot(times,x_time)

    '''
    for i in range(Nslices):
        plt.plot(freqAvg,Gxx[i,])

    plt.figure()
    plt.imshow(Gxx.T,aspect="auto",origin = "lower",extent=[0,T,0,fs/2])
    '''
    plt.show()


def spectrogram(x_time,fs,sliceLength,sync = 0,overlap=0,winType="uniform"):
    N = len(x_time)
    Nslices = int(N/sliceLength)
    T = Nslices*sliceLength/fs
    
    _, freqAvg, _, Gxx = spectroArray(x_time,fs,sliceLength,Nslices,sync,overlap=overlap,winType=winType)


    plt.imshow(Gxx.T, aspect="auto", origin="lower", extent=[0, T, 0, fs / 2])

def actual():
    filename="T4_C5_3L_dec4a.wav"

    path = foldername+filename
    print path

    fs , data = wavfile.read(path)

    N = len(data)




if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))

