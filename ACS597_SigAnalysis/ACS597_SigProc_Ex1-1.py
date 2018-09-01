# PSUACS
# ACS597_SigProc_Ex1-1
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/14/2018


import numpy as np
import matplotlib.pyplot as plt

def main(args):
    x_time = [4, 5, 2,-3, 7, 0,-2, 2]          # Random Sequence
    #x_time = [1, 1, 1, 1, 1, 1, 1, 1]          # Step Sequence
    #x_time = [1,-1, 1,-1, 1,-1, 1,-1]          # Nyquist Frequency Sequence
    #x_time = [1, 0, 1, 0, 1, 0,-1, 0]          # Nyquist Frequency/2 Sequence

    delT = 0.25                                 # Sample period

    N = len(x_time)                             # Number of samples
    fs = 1.0/delT                               # Sampling Frequency
    T = delT*N                                  # Duration of Sampling
    delF = 1/T                                  # Frequency bin width

    X_fft = np.fft.fft(x_time)                  ## Calculate FFT of x_time
    linSpec = X_fft*delT                        ## Calculate Linear Spectrum of x_time

    print "N: " + str(N)
    print u"\N{GREEK CAPITAL LETTER DELTA}" + "t: " + str(delT)
    print u"\N{GREEK CAPITAL LETTER DELTA}" + "f: " + str(delF)
    print "fs: " + str(fs)
    print "T: " + str(T)
    print 10*"-"
    print linSpec

    for i in range(len(linSpec)):
        if linSpec[i].imag < 0:
            op = ''
        else:
            op = '+'
        print '%.2f'%linSpec[i].real + op + '%.2f'%linSpec[i].imag +'j'

    for i in range((len(linSpec)/2)):
        if i > 0:
            im = (linSpec[i] + linSpec[-i]).imag        ## Compute the sum of the imaginary parts of complex conjugates
            print str(i) + " , " + str(len(linSpec) - i) + ": " + str(im)  ## and display values

    t = np.arange(0,N)*delT                     ## Initialize time array
    freq = np.arange(0,N)*delF                  ## Initialize frequency array

    plt.figure()
    plt.subplot(211)                            ## Plot x_time
    plt.plot(t,x_time)
    plt.xlim(t[0],t[-1])
    plt.title("x_time (Time Domain Sequence)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")


    plt.subplot(212)                            ## Plot the Linear Spectrum
    plt.plot(freq,abs(linSpec))
    plt.xlim(freq[0], freq[-1])
    plt.title("X_fft (FFT of x_fft)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.show()

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))