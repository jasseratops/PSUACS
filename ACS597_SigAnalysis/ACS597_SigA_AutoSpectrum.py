# PSUACS
# ACS597_SigA_AutoSpectrum
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/1/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp, genfromtxt
import sigA
import sounddevice as sd

foldername = "/Users/macbookpro/PycharmProjects/PSUACS/ACS597_SigAnalysis/TRAC1_and_3/"

def main(args):
    #grind('TRAC1_noise_time.csv')
    grind('TRAC3_sin100_time.csv')

    return 0



def grind(filename):
    path = foldername + filename
    dat = genfromtxt(path, delimiter=',')

    print np.shape(dat)
    print dat

    x_time = dat[:,1]
    t = dat[:,0]

    print t

    delT = t[1]-t[0]

    for i in range(len(t)):
        if i !=0:
            if (t[i]-t[i-1]) != (delT):
                sys.exit("delT is not constant, see t[" + str(i) + "]")

    fs = 1.0/float(delT)
    N = len(t)

    _,delF,T = sigA.param(N,fs)
    freq = sigA.freqVec(N,fs)

    sd.play(0.1*x_time,fs, blocking=True)

    GxxLim = (len(freq)/2) + 1

    rms,ms,m = sigA.rms(x_time)

    lsp = sigA.linSpec(x_time,fs)
    Sxx = sigA.dsSpec(x_time,fs)
    Gxx = sigA.ssSpec(x_time,fs)

    SxxRMS = sum(Sxx)*delF
    GxxRMS = sum(Gxx)*delF

    print "SxxRMS: " + str(SxxRMS)
    print "GxxRMS: " + str(GxxRMS)
    print "Amp_pk" + str(max(x_time))
    print "Amp_rms" + str(max(x_time)/np.sqrt(2.0))


    plt.figure()
    plt.plot(t,x_time)
    plt.title("Noise, Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")
    plt.xlim(t[0],t[-1])


    plt.figure()
    #plt.plot(freq[:posLim],abs(lsp[:posLim]),label="lsp")
    #plt.plot(freq[:posLim],Sxx[:posLim],label="Sxx")
    plt.plot(freq[:GxxLim],Gxx[:GxxLim])
    plt.title("Noise, Frequency Domain")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Amplitude [${WU^2}$/ Hz]")
    #plt.plot(freq[:len(Gxx)-1],Gxx)
    plt.xlim(freq[0],freq[GxxLim])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))