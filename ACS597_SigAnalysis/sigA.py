# PSUACS
# sigA
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/1/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import sys

import sounddevice as sd
import scipy.signal as sig

def param(N,fs,show=True):
    delT = 1/float(fs)
    delF = float(fs)/float(N)
    T = float(N)/float(fs)
    if show:
        print "N: " + str(N)
        print u"\N{GREEK CAPITAL LETTER DELTA}" + "t: " + str(delT)
        print u"\N{GREEK CAPITAL LETTER DELTA}" + "f: " + str(delF)
        print "fs: " + str(fs)
        print "T: " + str(T)
        print 10 * "-"
    return delT, delF, T


def timeVec(N, fs):
    delT,_,_ = param(N,fs,False)
    t = np.arange(0, N) * delT
    return t


def freqVec(N, fs):
    _, delF, _ = param(N, fs, False)
    f = np.arange(0, N) * delF
    return f

def check(x_time,fs):
    length = len(x_time)
    if length < 4:
        print "Array size: " + str(length)
        print "Array too small"

def linSpec(x_time,fs,winType="uniform"):
    N = len(x_time)
    delT, _, _ = param(N, fs, False)
    x_time = x_time*window(winType,N)
    return np.fft.fft(x_time)*delT

def timeSer(lsp, fs):
    _, delF, _ = param(N, fs, False)
    x_time  = np.fft.ifft(lsp)*delF
    return x_time

def rms(x_time,show=True):
    m_x_time = np.mean(x_time.real)
    ms_x_time = np.mean((x_time.real)**2)
    rms_x_time = np.sqrt(ms_x_time)

    if show:
        print "mean: " + str(m_x_time)
        print "ms: " + str(ms_x_time)
        print "rms: " + str(rms_x_time)

    return rms_x_time, ms_x_time, m_x_time

def PnN_LinSpec(x_time,fs,winType):
    N = len(x_time)
    ind = (N/2)+1               # Rounds down if length of x_time is odd
    lsp = linSpec(x_time,fs)
    pos_lsp = lsp[:ind]
    neg_lsp = lsp[ind:]

    return pos_lsp, neg_lsp

def dsSpec(x_time,fs,winType="uniform"):
    N = len(x_time)
    _, delF, _ = param(N, fs, False)
    lsp = linSpec(x_time,fs,winType)
    Sxx = (abs(lsp)**2)*delF
    return Sxx

def ssSpec(x_time,fs,winType="uniform"):
    Sxx = dsSpec(x_time,fs,winType)
    N = len(Sxx)

    Gxx = Sxx[0:(N/2)+1]
    odd = bool(len(Sxx)%2)

    for i in range(len(Gxx)):
        if (i != 0) or (i==(len(Gxx)-1) and odd):
            Gxx[i] = (Gxx[i])*2
    return Gxx

def spectroArray(x_time, fs, sliceLength, sync=0,overlap=0,winType="uniform"):
    # fix this
    # number of overlapping windows:
    overlap = np.abs(overlap)
    if overlap >= 1.0:
        sys.exit("overlap >= 1")

    N = len(x_time)
    m = int((N-int(overlap*sliceLength))/(sliceLength*(1-overlap)))
    Gxx = np.zeros((m,int(sliceLength/2)))

    for i in range(m):
        n = i * (int(sliceLength*(1-overlap))+sync)
        sliceEnd = int(n + sliceLength - 1)
        #print i
        #print sliceEnd
        Gxx[i,] = ssSpec(x_time[n:sliceEnd], fs,winType)

    #### Gxx Avg
    GxxAvg = np.mean(Gxx, axis=0)
    freqAvg = freqVec(sliceLength, fs)[:int(sliceLength/2)]
    _, delF_Avg, _ = param(sliceLength, fs, show=False)

    return GxxAvg, freqAvg, delF_Avg, Gxx

def spectrogram(x_time, fs, sliceLength, sync=0, overlap=0,color="jet", dB=True, winType="uniform", scale=True):
    N = len(x_time)
    Nslices = int(N / sliceLength)
    T = Nslices * sliceLength / float(fs)

    _, freqAvg, _, Gxx = spectroArray(x_time=x_time, fs=fs, sliceLength=sliceLength, sync=sync, overlap=overlap, winType=winType)

    GxxRef = 1.0                            # V^2/Hz
    Gxx_dB = 10 * np.log10(Gxx / GxxRef)

    ext = [0, T, 0, fs / 2]

    if dB:
        plt.imshow(Gxx_dB.T, aspect="auto", origin="lower", cmap=color, extent=ext)
    else:
        plt.imshow(Gxx.T, aspect="auto", origin="lower",cmap=color, extent=ext)
    if scale:
        plt.ylim(ext[1] + 1, ext[3] * 0.8)

def window(type, N):
    n = np.asfarray(np.arange(N))
    vec = n/N
    win = np.ones(N)
    if type == "uniform":
        win = np.ones(N)
    elif type == "hann":
        win = 1-cos(2*pi*vec)
    elif type == "flat top":
        win = 1-(1.93*cos(2*pi*vec))+(1.29*cos(4*pi*vec))\
              -(0.388*cos(6*pi*vec))+(0.322*cos(8*pi*vec))
    return win

def crossSpec(x_time,y_time,fs,winType="uniform"):
    N = len(x_time)
    if N != len(y_time):
        sys.exit("x_time and y_time must have the same length")
    _,delF,_ = param(N,fs,show=False)

    X = linSpec(x_time,fs,winType)
    Y = linSpec(y_time,fs,winType)

    Xconj_Y = np.conj(X) * Y
    S_XY = Xconj_Y*delF

    return S_XY

def play(x_time,fs):
    normFactor = 0.8/max(x_time)
    sdArray = (x_time*normFactor)
    sd.play(sdArray,fs)

def expAvging(x_time_sqrd, fs, timeConst):
    delT, _, _ = param(len(x_time_sqrd), fs, show=False)
    alpha = delT / timeConst
    a = [1., alpha - 1]
    b = [0, alpha]
    averaged = sig.lfilter(b, a, x_time_sqrd)

    return averaged

def fWarp(f,fs):
    f_w = np.tan(pi*f/fs)/(pi/fs)
    return f_w

def constQ(f0,Q,fs):
    w0 = 2*pi*f0
    num_s = np.array([w0/Q,0])
    den_s = np.array([1., w0/Q, w0**2])
    b,a = bilinearXform(num_s,den_s,fs)
    return b,a

def bilinearXform(nums, dens, fs):
    nn = len(nums)
    mm = len(dens)

    if nn > mm:
        sys.exit("Order of numerator cannot exceed order of denominator")
    elif nn < mm:
        # print "padding"
        padz = np.zeros(mm - nn)
        nums = np.append(padz, nums)

    rootv = -np.ones(mm - 1)
    numz = np.zeros(mm)
    denz = numz
    fsfact = 1.

    for i in range(mm):
        basepoly = np.poly(rootv) * fsfact
        numz = numz + basepoly * nums[mm - i - 1]
        denz = denz + basepoly * dens[mm - i - 1]

        if i < (mm - 1):
            fsfact = fsfact * 2 * fs
            rootv[i] += 2
    print numz
    print denz

    numz /= denz[0]
    denz /= denz[0]

    return numz, denz