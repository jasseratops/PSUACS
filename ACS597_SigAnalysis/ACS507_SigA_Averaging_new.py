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

#folder = "C:/Users/alshehrj/PycharmProjects/PSUACS/ACS597_SigAnalysis/"
folder = "/Users/macbookpro/PycharmProjects/PSUACS/ACS597_SigAnalysis/HW3/"


def ssSpecAvg(x_time, fs, N, Nl, syncStep=0):
    Gxx = np.zeros((Nl, N / 2))

    for i in range(Nl):
        n = i * (N+syncStep)
        Gxx[i] = sigA.ssSpec(x_time[n:n + N - 1], fs)
    #### Gxx Avg
    GxxAvg = np.mean(Gxx, axis=0)

    freqAvg = sigA.freqVec(N, fs)
    _, delF_Avg, _ = sigA.param(N, fs, show=False)

    return GxxAvg, freqAvg, delF_Avg, Gxx


def lSpecAvg(x_time, fs, N, Nl, syncStep=0):
    lsp = np.zeros((Nl, N),dtype=complex)

    for i in range(Nl):
        n = i * (N+syncStep)
        lsp[i] = sigA.linSpec(x_time[n:n + N], fs)

    #### lSpec Avg
    lspAvg = np.mean(lsp, axis=0)

    freqAvg = sigA.freqVec(N, fs)
    _, delF_lspAvg, _ = sigA.param(N, fs, show=False)

    return lspAvg, freqAvg, delF_lspAvg, lsp

def timeAvg(x_time,fs, N, Nl,syncStep=0):
    x_n = np.zeros((Nl, N))

    for i in range(Nl):
        n = i * (N + syncStep)
        x_n[i] = x_time[n:n + N]

    x_n_Avg = np.mean(x_n, axis=0)

    times = sigA.time(N, fs)
    delT_Avg, _, _ = sigA.param(N, fs, show=False)

    return x_n_Avg, times, delT_Avg, x_n



def main(args):
    partA()
    partB()

def partA():
    filename = "S_plus_N_20.wav"
    path = folder+filename
    print path

    fs , data = wavfile.read(path)

    Nwav = len(data)

    print fs
    print data
    print Nwav
    print 10*"-"

    t = sigA.timeVec(Nwav,fs)


    #### 200 runs of Gxx
    N = 1024
    Nl = 200

    # Gxx Avg
    GxxAvg, freqAvg, delF_Avg, Gxx = ssSpecAvg(data,fs,N,Nl)


    # Gxx Long
    NTot = Nl * N
    GxxTot = np.zeros(NTot)
    GxxTot = sigA.ssSpec(data[:NTot], fs)
    freqTot = sigA.freqVec(NTot, fs)
    _, delF_Tot, _ = sigA.param(NTot, fs,show=False)

    rmsAvg = np.sum(GxxAvg) * delF_Avg
    rmsLong = np.sum(GxxTot) * delF_Tot


    print "rmsAvg: " + str(rmsAvg)
    print "rmsLong: " + str(rmsLong)


    plt.figure()
    plt.plot(t,data)

    plt.figure()
    plt.plot(freqAvg[:len(GxxAvg)],GxxAvg)
    #plt.plot(freqAvg[:len(GxxAvg)],Gxx[13,])
    plt.title("GxxAvg")


    plt.figure()
    plt.plot(freqTot[:len(GxxTot)],GxxTot)
    plt.title("GxxTot")
    plt.show()

    print "finished part A"

    return 0




def partB():
    filename = "P_plus_N_10.wav"
    path = folder+filename
    eventPeriod= 1111

    fs , data = wavfile.read(path)

    Nwav = len(data)

    print fs
    print data
    print Nwav
    print 10*"-"

    t = sigA.timeVec(Nwav,fs)



    #### 200 runs of Gxx
    N = 1024
    Nl = 200
    syncStep = eventPeriod - 1024

    # Gxx Avg
    GxxAvg, freqAvg, delF_Avg, Gxx = ssSpecAvg(data,fs,N,Nl)


    # Gxx Long
    NTot = Nl * N
    GxxTot = np.zeros(NTot)
    GxxTot = sigA.ssSpec(data[:NTot], fs)
    freqTot = sigA.freqVec(NTot, fs)
    _, delF_Tot, _ = sigA.param(NTot, fs,show=False)

    lspAvg, freqLsAvg,delFlSpecAvg, lSpec = lSpecAvg(data,fs,N,Nl,syncStep)


    rmsAvg = np.sum(GxxAvg) * delF_Avg
    rmsLong = np.sum(GxxTot) * delF_Tot


    print "rmsAvg: " + str(rmsAvg)
    print "rmsLong: " + str(rmsLong)


    plt.figure()
    plt.plot(t,data)

    plt.figure()
    plt.plot(freqAvg[:len(GxxAvg)],GxxAvg)
    #plt.plot(freqAvg[:len(GxxAvg)],Gxx[13,])
    plt.title("GxxAvg")


    plt.figure()
    plt.plot(freqTot[:len(GxxTot)],GxxTot)
    plt.title("GxxTot")
    plt.show()

    print "finished part B"

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))