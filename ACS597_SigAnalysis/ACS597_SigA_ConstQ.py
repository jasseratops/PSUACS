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
    T = 20.
    N = int(T*fs)

    f1 = f0-(bw/2)
    f2 = f0+(bw/2)
    f0_warp = fWarp(f0,fs)
    f1_warp = fWarp(f1,fs)
    f2_warp = fWarp(f2,fs)
    print f0_warp
    print f1_warp
    print f2_warp
    bw_warp = f2_warp-f1_warp
    Q = f0_warp/bw_warp
    print Q

    b, a = constQ(f0,Q,fs,N)

    delT,_,_=sigA.param(N,fs)

    w, h = sig.freqz(b,a)
    f = w*fs/(2*pi)
    m = np.abs(h)
    p = np.angle(h)

    imp_train = np.zeros(N)
    imp_train[0] = 1./delT

    impResp = sig.lfilter(b,a,imp_train)

    print b,a

    times = sigA.timeVec(N,fs)
    sineWave = sin(2*np.pi*f0*times)
    sineResp = sig.lfilter(b,a,sineWave)

    sineWave_sqrd = sineWave**2

    smooth = expAvging(sineWave_sqrd,fs,timeConst=1.)

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

    '''
    plt.figure()

    plt.plot(times,sineResp)
    plt.plot(times,sineWave)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [Pa]")
    plt.title("Sine Response of Constant-Q Filter (f0 = 200)")
    '''
    plt.figure()
    plt.plot(times,sineWave)
    plt.plot(times,sineWave_sqrd)
    plt.plot(times,smooth)

    plt.show()
    return 0

def expAvging(x_time_sqrd,fs,timeConst):
    delT,_,_ = sigA.param(len(x_time_sqrd),fs,show=False)
    alpha = delT/timeConst
    a = [1.,alpha-1]
    b = [0,alpha]
    averaged = sig.lfilter(b,a,x_time_sqrd)

    return averaged

def fWarp(f,fs):
    f_w = np.tan(pi*f/fs)/(pi/fs)
    return f_w

def constQ(f0,Q,fs,N):
    w0 = 2*pi*f0
    num_s = np.array([w0/Q,0])
    den_s = np.array([1., w0/Q, w0**2])
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