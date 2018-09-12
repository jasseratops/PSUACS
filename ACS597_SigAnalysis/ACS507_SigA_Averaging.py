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

def main(args):
    function("S_plus_N_20.wav")
    #function("P_plus_N_10.wav")

def function(filename):
    path = folder+filename
    print path

    #data = wave.open(path,'r')

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

    Gxx = np.zeros((Nl,N/2))

    for i in range(200):
        n = i*N
        Gxx[i] = sigA.ssSpec(data[n:n+N-1],fs)

    freq = sigA.freqVec(N, fs)

    plt.figure()
    for i in range(Nl):
        plt.plot(freq[:np.shape(Gxx)[1]],Gxx[i,])

    #### Gxx Avg
    GxxAvg = np.mean(Gxx,axis=0)

    _, delF_Avg,_ = sigA.param(N,fs,show=True)

    rmsAvg = np.sum(GxxAvg)*delF_Avg
    print "rmsAvg: " + str(rmsAvg)

    plt.figure()
    plt.plot(freq[:np.shape(Gxx)[1]],GxxAvg)

    #### Gxx Long

    NTot= Nl*N

    GxxTot = np.zeros(NTot)
    print np.shape(GxxTot)

    GxxTot = sigA.ssSpec(data[:NTot],fs)
    freqTot = sigA.freqVec(NTot,fs)

    _, delF_Tot ,_ = sigA.param(NTot,fs,show=True)

    rmsLong = np.sum(GxxTot)*delF_Tot
    print "rmsLong: " + str(rmsLong)

    plt.figure()
    plt.plot(freqTot[:len(GxxTot)],GxxTot)
    plt.show()

    '''
    print fs
    t = sigA.timeVec(len(data),fs)

    sd.play(data*0.5,fs,blocking=True)

    plt.figure()
    plt.plot(t,data)
    '''

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))