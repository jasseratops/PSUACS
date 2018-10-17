# PSUACS
# ACS597_SigA_ConstQ
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/16/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import scipy.signal as sig
import sigA

def main(args):
    f0 = 200.
    bw = 0.1


    fs = 1000.
    T = 20
    N = int(T*fs)

    b, a = constQ(f0,bw,fs,N)

    delT,_,_=sigA.param(N,fs)

    w, h = sig.freqz(b,a)
    f = w*fs/(2*pi)
    m = np.abs(h)
    m/=max(m)
    p = np.angle(h)

    imp_train = np.zeros(N)
    imp_train[0] = 1./delT

    impResp = sig.lfilter(b,a,imp_train)

    print b,a

    times = sigA.timeVec(N,fs)
    sineWave = sin(2*np.pi*200*times)
    sineResp = sig.lfilter(b,a,sineWave)

    plt.figure()
    plt.semilogy(f,m)
    plt.axhline(0.707*max(m),color="black")
    plt.xlim(f[0],f[-1])
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Response Magnitude [Pa/Hz]")
    plt.title("Frequency Response of Constant-Q Filter (f0 = 200)")


    plt.figure()
    plt.plot(times,impResp)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [Pa]")
    plt.title("Impulse Response of Constant-Q Filter (f0 = 200)")



    plt.figure()
    plt.plot(times,sineResp)
    plt.plot(times,sineWave)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [Pa]")
    plt.title("Sine Response of Constant-Q Filter (f0 = 200)")
    plt.show()

    return 0

def expAvging():
    avg =
    return avg

def constQ(f0,bw,fs,N):
    f1 = f0-(bw/2)
    f2 = f0+(bw/2)
    delT,delF, _ = sigA.param(N=N,fs=fs,show=False)

    f0_warp = (np.tan(pi*(f0/fs))/(pi/fs))
    f1_warp = np.tan(pi * (f1 / fs)) / (pi / fs)
    f2_warp = np.tan(pi * (f2 / fs)) / (pi / fs)
    print f0_warp
    print f1_warp
    print f2_warp
    bw_warp = f2_warp-f1_warp
    Q = f0_warp/bw_warp
    w0 = 2*pi*f0_warp

    num_s = [w0/Q,0]
    den_s = [1, w0/Q, w0**2]
    b,a = bilinearXform(num_s,den_s,fs)
    return b,a

def bilinearXform(nums,dens,fs):
    nn = len(nums)
    mm = len(dens)

    if nn > mm:
        sys.exit("Order of numerator cannot exceed order of denominator")
    elif nn < mm:
        #print "padding"
        padz = np.zeros(mm-nn)
        nums = np.append(padz,nums)

    rootv = -np.ones(mm-1)
    numz = np.zeros(mm)
    denz = numz
    fsfact = 1.

    for i in range(mm):
        basepoly = np.poly(rootv)*fsfact
        numz = numz + basepoly*nums[mm-i-1]
        denz = denz + basepoly*dens[mm-i-1]

        if i<(mm-1):
            fsfact=fsfact*2*fs
            rootv[i] += 2
    print numz
    print denz

    numz /= denz[0]
    denz /= denz[0]

    return numz, denz

if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))