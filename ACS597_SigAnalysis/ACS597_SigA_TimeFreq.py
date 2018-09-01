# PSUACS
# ACS597_SigA_TimeFreq
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/30/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import random as rn
import time
import sounddevice as sd

def main(args):
    rn.seed(5)

    N = 48000
    fs = 24000.00
    T = N/fs
    delF = 1/T
    delT = T / N

    t = np.arange(0,N)*delT                     ## Initialize time array
    freq = np.arange(0,N)*delF                  ## Initialize frequency array

    print "N: " + str(N)
    print u"\N{GREEK CAPITAL LETTER DELTA}" + "t: " + str(delT)
    print u"\N{GREEK CAPITAL LETTER DELTA}" + "f: " + str(delF)
    print "fs: " + str(fs)
    print "T: " + str(T)
    print 10*"-"


    pos_lsp = np.zeros(N/2+1,dtype=complex)
    neg_lsp = np.zeros(N/2-1,dtype=complex)
    phase = np.zeros(N)



    print "Creating Positive LSP"

    for i in range(len(pos_lsp)):
        #print 10*"-"
        #print "N: " + str(i)


        if i == 0 or i == N/2:
            #A = 0.0
            A = 1
            phi = 0
        else:
            A = 1.0
            #A = 1.0/(float(i))
            phi = (rn.random())*2*pi
            #phi = 0

        pos_lsp[i] = A*exp(1j*phi)

        '''
        if np.angle(pos_lsp[i]) < 0:
            print "if: " + str(np.angle(pos_lsp[i])+(2*pi))
        else:
            print "else: " + str(np.angle(pos_lsp[i]))
        '''

    print 10*"-"
    print "Creating Negative LSP"
    neg_lsp = np.conj(pos_lsp[1:-1])[::-1]      #taking a subset of pos_lsp, getting the complex conjugate of each element, and reversing the array

    lsp = np.append(pos_lsp,neg_lsp)

    for i in range(N):
        ang = np.angle(lsp[i])
        if ang < 0:
            phase[i] = ang + (2*pi)
        else:
            phase[i] = ang


    print np.shape(lsp)
    print pos_lsp
    print neg_lsp
    print lsp
    print lsp[N/2]


    x_ifft = np.fft.ifft(lsp)
    x_time = (x_ifft*(1/delT)).real
    print "x_ifft"
    print x_ifft

    m_x_time = np.mean(x_time)
    ms_x_time = np.mean(x_time**2)
    rms_x_time = np.sqrt(ms_x_time)
    ratPtoRMS = max(abs(x_time))/rms_x_time

    print "mean: " + str(m_x_time)
    print "ms: " + str(ms_x_time)
    print "rms: " + str(rms_x_time)
    print "max: " + str(max(x_time))
    print "max/rms: " + str(ratPtoRMS)

    sdArray = (x_time/(max(abs(x_time))))
    sdArray = np.append(sdArray,sdArray)

    print "Playing/recording Sound"
    myRec = sd.playrec(sdArray,fs,channels=1)
    time.sleep(4)
    trans_skip = 6020

    rec_x_time = myRec[trans_skip:(trans_skip+48000)]
    rec_x_time = (max(x_time)/(max(myRec)))*(rec_x_time - np.mean(rec_x_time))
    print np.shape(rec_x_time)

    rec_fft = np.fft.fft(rec_x_time)
    rec_lsp = rec_fft*delT

    plt.figure()
    plt.plot(myRec)

    plt.figure()
    plt.plot(t,rec_x_time)
    plt.title("Recorded Noise")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")
    plt.xlim(0,2)

    plt.figure()
    plt.plot(freq,abs(rec_lsp))
    plt.title("Linear Spectrum of Recorded Noise")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [WU]")
    plt.xlim(0,12000)

    plt.figure()
    plt.subplot(211)
    plt.plot(freq, abs(lsp))
    plt.title("Frequency Response")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [DU]")

    plt.subplot(212)
    #plt.semilogx(phase)
    plt.plot(freq, np.angle(lsp))
    plt.title("Phase Response")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [rad]")

    plt.figure()
    plt.plot(t, x_time)
    plt.title("Generated Noise (Time-Domain)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [WU]")
    plt.xlim(-0.01,max(t))

    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))