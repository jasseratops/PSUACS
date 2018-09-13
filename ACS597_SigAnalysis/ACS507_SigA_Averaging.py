# PSUACS
# ACS507_SigA_Averaging
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/11/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
from scipy.io import wavfile
import wave
import sounddevice as sd
import sigA

folder = "C:/Users/alshehrj/PycharmProjects/PSUACS/ACS597_SigAnalysis/"


def ssSpecAvg(x_time, fs, N, Nl, syncStep):
    Gxx = np.zeros((Nl, N / 2))

    for i in range(Nl):
        n = i * (N+syncStep)
        Gxx[i] = sigA.ssSpec(x_time[n:n + N - 1], fs)

    freqAvg = sigA.freqVec(N, fs)
    #### Gxx Avg
    GxxAvg = np.mean(Gxx, axis=0)

    _, delF_Avg, _ = sigA.param(N, fs, show=True)

    return GxxAvg, freqAvg, delF_Avg, Gxx

def lSpecAvg(x_time, fs, N, Nl, syncStep):
    lSpec = np.zeros((Nl, N / 2))

    for i in range(Nl):
        n = i * (N+syncStep)
        lSpec[i] = sigA.linSpec(x_time[n:n + N - 1], fs)

    freqAvg = sigA.freqVec(N, fs)
    #### lSpec Avg
    lSpecAvg = np.mean(lSpec, axis=0)

    _, delF_Avg, _ = sigA.param(N, fs, show=True)

    return lSpecAvg, freqAvg, delF_Avg, lSpec

def main(args):
    #GxxComp("S_plus_N_20.wav",0)
    GxxComp("P_plus_N_10.wav",87)



def GxxComp(filename,syncStep):
    path = folder+filename
    print path


    fs , data = wavfile.read(path)

    Nwav = len(data)

    print fs
    print data
    print Nwav
    print 10*"-"

    t = sigA.timeVec(Nwav,fs)

    plt.figure()
    plt.plot(t,data)

    #### 200 runs of Gxx
    N = 1024
    Nl = 200

    GxxAvg, freqAvg, delF_Avg, Gxx = ssSpecAvg(data,fs,N,Nl,syncStep)


    plt.figure()
    plt.plot(freqAvg[:len(GxxAvg)],GxxAvg)
    #plt.plot(freqAvg[:len(GxxAvg)],Gxx[13,])

    rmsAvg = np.sum(GxxAvg) * delF_Avg

    print "rmsAvg: " + str(rmsAvg)


    #### Gxx Long

    NTot= Nl*N

    GxxTot = np.zeros(NTot)

    GxxTot = sigA.ssSpec(data[:NTot],fs)
    freqTot = sigA.freqVec(NTot,fs)

    _, delF_Tot ,_ = sigA.param(NTot,fs,show=True)

    rmsLong = np.sum(GxxTot)*delF_Tot
    print "rmsLong: " + str(rmsLong)

    plt.figure()
    plt.plot(freqTot[:len(GxxTot)],GxxTot)
    plt.show()

    return 0

def timeAvg(filename,syncStep):



if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))