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
    fs = 1000.
    N = 2048

    delT,delF, T = sigA.param(N=N,fs=fs)

    f0_warp = np.tan(pi*(f0/fs))/(pi/fs)

    bw = 0.1
    Q = f0_warp/bw
    w0 = 2*pi*f0_warp

    num_s = [w0/Q,0]
    den_s = [1, w0/Q, w0**2]
    print den_s
    b,a = bilinearXform(num_s,den_s,fs)


    w, h = sig.freqz(b,a)
    f = w*fs/(2*pi)
    m = np.abs(h)
    m/=max(m)
    p = np.angle(h)

    imp_train = np.zeros(N)
    imp_train[0] = 1/delT

    impResp = sig.lfilter(b,a,imp_train)
    t , imp2 = sig.dimpulse([b,a,delT],n=N)
    print "hi"
    print imp2[0].T
    imp2 = np.reshape(imp2[0], (len(imp2[0]),))

    print np.shape(imp2)


    plt.figure()
    plt.plot(f,m)
    plt.axhline(0.707)

    plt.figure()
    plt.stem(impResp)

    plt.figure()
    plt.stem(imp2)
    plt.show()

    return 0


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

        print nums
        print numz
        print denz
        print basepoly
        print mm - i - 1

    numz /= numz[0]
    denz /= denz[0]

    return numz, denz

if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))