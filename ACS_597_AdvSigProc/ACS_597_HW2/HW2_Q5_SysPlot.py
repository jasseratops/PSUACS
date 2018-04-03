# JAscripts
# HW2_Q5_SysPlot
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/11/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import scipy.signal as sig


def main(args):

    a = [1,0,0.9]
    b = [0.3,0.6,0.3]
    #x = np.ones(128)
    x = sig.unit_impulse(128)
    print x

    yOx = sig.lfilter(b,a,x)
    yOxNORM = yOx
    yOxIMPZ = impz(b,a)
    w,yOxFREQZ = mfreqz(b,a)
    z,yOxFREQZ2 = mfreqz(yOxIMPZ[:79],1)

    plt.figure()
    plt.title("Impulse Response Using Impulse Sequence")
    plt.stem(yOxNORM)

    plt.figure()
    plt.stem(yOxIMPZ)

    plt.figure()
    plt.plot(w,yOxFREQZ,label="Freq. Response")
    plt.plot(w,yOxFREQZ2, label="Freq. Response using Imp. Response")
    #plt.stem(yOx)

    plt.show()

    return 0

def mfreqz(b,a=1):
    w,h = sig.freqz(b,a)
    h_dB = 20 * np.log10(abs(h))
    '''
    plt.figure()
    #plt.subplot(211)
    plt.plot(w/max(w),h_dB)
    #plt.ylim(-150, 5)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')

    plt.subplot(212)
    h_Phase = np.unwrap(np.arctan2(h.imag,h.real))
    plt.plot(w/max(w),h_Phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency ($\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)
    '''
    return w,h_dB

def impz(b,a=1,n=50):
    impulse = sig.unit_impulse(n)
    #x = np.arange(0,n)
    response = sig.lfilter(b,a,impulse)
    maxResp = max(response)
    gauge = maxResp*(5E-5)
    more = 0
    x = 0

    for i in range(len(response)):
        if response[i] <= gauge:
            more += 1
            if x == 0:
                x = i

	if more == 0:
		impz(b,a,2*n)




'''
def impz(b,a=1):
    impulse = np.repeat(0.,50); impulse[0] =1.
    x = np.arange(0,50)
    response = sig.lfilter(b,a,impulse)
    plt.figure()
    #plt.subplot(211)
    plt.stem(x, response)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Impulse response')

    plt.subplot(212)
    step = np.cumsum(response)
    plt.stem(x, step)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Step response')
    plt.subplots_adjust(hspace=0.5)
    return response
    '''

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))