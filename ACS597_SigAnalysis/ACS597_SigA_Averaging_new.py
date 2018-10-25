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
#folder = "/Users/macbookpro/PycharmProjects/PSUACS/ACS597_SigAnalysis/HW3/"


def ssSpecAvg(x_time, fs, N, Nl, sync=0):
    Gxx = np.zeros((Nl, N / 2))
    for i in range(Nl):
        n = i * (N+sync)
        Gxx[i] = sigA.ssSpec(x_time[n:n + N - 1], fs)
    #### Gxx Avg
    GxxAvg = np.mean(Gxx, axis=0)
    freqAvg = sigA.freqVec(N, fs)
    _, delF_Avg, _ = sigA.param(N, fs, show=False)

    return GxxAvg, freqAvg, delF_Avg, Gxx

def lSpecAvg(x_time, fs, N, Nl, sync=0):
    lsp = np.zeros((Nl, N))
    for i in range(Nl):
        n = i * (N+sync)
        lsp[i] = abs(sigA.linSpec(x_time[n:n + N], fs))
    #### lSpec Avg
    lspAvg = np.mean(lsp, axis=0)
    freqAvg = sigA.freqVec(N, fs)
    _, delF_lspAvg, _ = sigA.param(N, fs, show=False)

    return lspAvg, freqAvg, delF_lspAvg, lsp

def timeAvg(x_time,fs, N, Nl,sync=0):
    x_n = np.zeros((Nl, N))
    for i in range(Nl):
        n = i * (N + sync)
        x_n[i] = x_time[n:n + N]
    x_n_Avg = np.mean(x_n, axis=0)
    times = sigA.timeVec(N, fs)
    delT_Avg, _, _ = sigA.param(N, fs, show=False)

    return x_n_Avg, times, delT_Avg, x_n


def main(args):
    partA()
    #partB()

def partA():
    filename = "S_plus_N_20.wav"
    path = folder+filename
    print path

    fs , data = wavfile.read(path)

    Nwav = len(data)

    if data.dtype != np.float32:
        print "Converting from " + str(data.dtype) + " to float32"
        data = data.astype(np.float32)
        data = data/32768.0

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


    print "Sine wave freq (GxxAvg): " + str(freqAvg[np.argmax(GxxAvg)])
    print "Max(GxxAvg): " + str(max(GxxAvg))
    print "Sine wave freq (GxxLong): " + str(freqTot[np.argmax(GxxTot)])
    print "Max(GxxLong): " + str(max(GxxTot))

    rmsAvg = np.sum(GxxAvg) * delF_Avg
    rmsLong = np.sum(GxxTot) * delF_Tot

    print "rmsAvg: " + str(rmsAvg)
    print "rmsLong: " + str(rmsLong)

    plt.figure()
    plt.plot(t,data)
    plt.title("S_plus_N Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")

    plt.figure()
    plt.plot(freqAvg[:len(GxxAvg)],GxxAvg,)
    plt.xlim(freqAvg[0],freqAvg[len(GxxAvg)])
    plt.title("$\overline{G_{XX}}$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Amplitude [${WU^2}$/ Hz]")


    plt.figure()
    plt.plot(freqAvg[:len(GxxAvg)],GxxAvg,label="$\overline{G_{XX}}$")
    plt.plot(freqAvg[:len(GxxAvg)],Gxx[13,],label="${G_{XX}[13]}$")
    plt.xlim(freqAvg[0], freqAvg[len(GxxAvg)])
    plt.title("$\overline{G_{XX}}$ vs ${G_{XX}[13]}$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Amplitude [${WU^2}$/ Hz]")
    plt.legend()

    plt.figure()
    plt.plot(freqTot[:len(GxxTot)],GxxTot)
    plt.title("$\overline{G_{Long}}$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Amplitude [${WU^2}$/ Hz]")
    plt.show()

    print "finished part A"

    return 0


def partB():
    filename = "P_plus_N_10.wav"
    path = folder+filename
    eventPeriod= 1111

    fs , data = wavfile.read(path)

    Nwav = len(data)
    print data.dtype

    if data.dtype != np.float32:
        print "Converting from " + str(data.dtype) + " to float32"
        data = data.astype(np.float32)
        data = data/32768.0

    print fs
    print data
    print Nwav
    print 10*"-"

    t = sigA.timeVec(Nwav,fs)


    #### 200 runs of Gxx
    N = 1024
    Nl = 200
    syncStep = eventPeriod - N

    # Gxx Avg
    GxxAvg, freqAvg, delF_Avg, Gxx = ssSpecAvg(data,fs,N,Nl)
    GxxAvgSync, freqAvgSync, delF_AvgSync, GxxSync = ssSpecAvg(data,fs,N,Nl,syncStep)

    LSP_Avg, freq_LSP_Avg, delF_LSP_Avg, LSP = lSpecAvg(data,fs,N,Nl)
    LSP_AvgSync, freq_LSP_AvgSync, delF_LSP_AvgSync, LSP_Sync = lSpecAvg(data,fs,N,Nl,syncStep)

    x_time_avg,tAvg, delT ,x_time  = timeAvg(data,fs,N,Nl,syncStep)

    LSP_Xavg = abs(sigA.linSpec(x_time_avg,fs))

    plt.figure()
    plt.plot(freqAvg[:len(GxxAvg)],GxxAvg,label="$\overline{G_{XX}}$")
    plt.plot(freqAvg[:len(GxxAvg)],GxxAvgSync,label="$\overline{G_{XXsync}}$")
    plt.plot(freqAvg[:len(GxxAvg)],GxxSync[13],label="${G_{XX}}$[13]")
    plt.xlim(freqAvg[0],freqAvg[len(GxxAvg)])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [${WU^2}$/ Hz]")
    plt.legend()

    plt.figure()
    plt.plot(freq_LSP_Avg[:len(LSP_Avg)/2],LSP_Avg[:len(LSP_Avg)/2],label="$\overline{LSP}$")
    plt.plot(freq_LSP_Avg[:len(LSP_Avg)/2],LSP_AvgSync[:len(LSP_Avg)/2],label="$\overline{LSP_{Sync}}$")
    plt.plot(freq_LSP_Avg[:len(LSP_Avg)/2],LSP_Xavg[:len(LSP_Avg)/2],label="LSP($\overline{x_{n}}$)")
    plt.xlim(freq_LSP_Avg[0],freq_LSP_Avg[len(GxxAvg)])
    plt.title("Linear Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [WU/Hz]")


    plt.legend()

    plt.figure()
    plt.plot(tAvg,x_time_avg,color="r",linewidth="5", label="$\overline{x_{n}}$")
    plt.plot(tAvg,x_time[13],color="g",label="${x_{n}}$[13]")
    plt.xlim(tAvg[0],tAvg[-1])
    plt.title("Pulse Time Series")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")
    plt.legend()

    plt.show()

    print "finished part B"

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))