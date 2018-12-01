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

    Gxx = Sxx[0:(N/2)+1] * 2
    odd = bool(len(Sxx)%2)

    for i in range(len(Gxx)):
        if not((i != 0) or (i==(len(Gxx)-1) and odd)):
            Gxx[i] = (Gxx[i])*0.5
    return Gxx

def timeAvg(x_time, fs, recLength, Nrecs, sync=0):
    if np.shape(sync)==():
        sync = np.zeros(Nrecs)+sync
    elif len(sync)!=Nrecs:
        sys.exit("sync must be single int, or same shape as Nrecs")

    x_n = np.zeros((Nrecs, recLength))
    for i in range(Nrecs):
        n = i * (recLength + int(sync[i]))
        x_n[i] = x_time[n:n + recLength]
    x_n_Avg = np.mean(x_n, axis=0)
    times = timeVec(recLength, fs)
    delT_Avg, _, _ = param(recLength, fs, show=False)

    return x_n_Avg, times, delT_Avg, x_n

def spectroArray(x_time, fs, sliceLength, sync=0,overlap=0,winType="uniform"):
    # fix this
    # number of overlapping windows:
    overlap = np.abs(overlap)
    if overlap >= 1.0:
        sys.exit("overlap >= 1")

    N = len(x_time)
    m = int((N-int(overlap*sliceLength))/(sliceLength*(1-overlap)))
    Gxx = np.zeros((m,int(sliceLength/2)))

    x_ms = np.zeros(m)


    for i in range(m):
        n = i * (int(sliceLength*(1-overlap))+sync)
        sliceEnd = int(n + sliceLength - 1)
        #print i
        #print sliceEnd
        Gxx[i,] = ssSpec(x_time[n:sliceEnd], fs,winType)
        x_ms[i] = rms(x_time[n:sliceEnd]*window(winType,int(sliceLength-1)),show=False)[1]

    #### Gxx Avg
    GxxAvg = np.mean(Gxx, axis=0)
    freqAvg = freqVec(sliceLength, fs)[:int(sliceLength/2)]
    _, delF_Avg, _ = param(sliceLength, fs, show=False)

    return GxxAvg, freqAvg, delF_Avg, Gxx#, x_ms

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

def crossCorrSpectrogram(x_time,y_time,fs,sliceLength,sync=0,overlap=0,color="jet",norm=True,pad=True):
    N = len(x_time)
    Nslices = int(N / sliceLength)
    T = Nslices * sliceLength / float(fs)

    _, timeShift, R_XY, C_XY = crossCorrArray(x_time,y_time,fs,sliceLength,pad,sync,overlap)

    TD = np.zeros(Nslices)

    for i in range(Nslices):
        TD[i] = timeShift[np.argmax(C_XY[i,:])]
    ext = [0, T, timeShift[0], timeShift[-1]]

    if norm:
        plt.imshow(np.abs(C_XY).T, aspect="auto", origin="lower", cmap=color, extent=ext)
    else:
        plt.imshow(abs(R_XY).T, aspect="auto", origin="lower", cmap=color, extent=ext)

    return TD

def crossCorrArray(x_time,y_time, fs, sliceLength,pad=True, sync=0,overlap=0):
    overlap = np.abs(overlap)
    if overlap >= 1.0:
        sys.exit("overlap >= 1")

    N = len(x_time)
    m = int((N - int(overlap * sliceLength)) / (sliceLength * (1 - overlap)))
    if pad:
        factor = 2
    else:
        factor = 1
    R_XY = np.zeros((m,factor*(sliceLength-1)), dtype=complex)
    C_XY = np.zeros((m,factor*(sliceLength-1)), dtype=complex)

    for i in range(m):
        n = i * (int(sliceLength*(1-overlap))+sync)
        sliceEnd = int(n + sliceLength - 1)
        x = x_time[n:sliceEnd]
        y = y_time[n:sliceEnd]

        if pad:
            x = np.append(x,np.zeros(len(x)))
            y = np.append(y,np.zeros(len(y)))
        x_RMS = rms(x,show=False)[0]
        y_RMS = rms(y,show=False)[0]
        R_XY[i,], timeShift = crossCorr(x,y, fs)
        C_XY[i,] = R_XY[i,]/(x_RMS*y_RMS)

    #### R_XY Avg
    R_XYavg = np.mean(R_XY, axis=0)

    return R_XYavg, timeShift, R_XY, C_XY


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

    Xconj_Y = (np.conj(X) * Y)
    S_XY = Xconj_Y*delF

    return S_XY

def ssCrossSpec(x_time,y_time,fs,winType="uniform"):
    S_XY = crossSpec(x_time,y_time,fs,winType)
    N = len(S_XY)

    G_XY = S_XY[0:(N / 2) + 1]*2
    odd = bool(len(S_XY) % 2)

    for i in range(len(G_XY)):
        if not((i != 0) or (i == (len(G_XY) - 1) and odd)):
            G_XY[i] = (G_XY[i]) * 0.5
    return G_XY

def crossSpectroArray(x_time,y_time, fs, sliceLength, sync=0,overlap=0,winType="uniform"):
    overlap = np.abs(overlap)
    if overlap >= 1.0:
        sys.exit("overlap >= 100%")

    N = len(x_time)
    m = int((N-int(overlap*sliceLength))/(sliceLength*(1-overlap)))
    G_XY = np.zeros((m,int(sliceLength/2)),dtype=complex)

    for i in range(m):
        n = i * (int(sliceLength*(1-overlap))+sync)
        sliceEnd = int(n + sliceLength - 1)
        #print i
        #print sliceEnd
        G_XY[i,] = ssCrossSpec(x_time[n:sliceEnd],y_time[n:sliceEnd], fs,winType)

    #### G_XY Avg
    G_XYavg = np.mean(G_XY, axis=0)
    freqAvg = freqVec(sliceLength, fs)[:int(sliceLength/2)]
    _, delF_Avg, _ = param(sliceLength, fs, show=False)

    return G_XYavg, freqAvg, delF_Avg, G_XY

def autocor(x_time,fs):
    N= len(x_time)
    delT,_,T= param(N,fs,show=False)
    S_XX = dsSpec(x_time,fs)
    R_XX = np.fft.ifft(S_XX)/delT
    sub = 0
    if N%2:
        N+=1
        sub =1

    R_XX = np.concatenate((R_XX[(N/2):],R_XX[0:N/2]))
    times = timeVec(N,fs)
    timeShift = np.concatenate((-1*times[1:(N/2)+1-sub][::-1],times[0:(N/2)]))

    return R_XX, timeShift

def crossCorr(x_time,y_time,fs):
    N = len(x_time)
    delT,_,_ = param(N,fs,show=False)
    S_XY = crossSpec(x_time,y_time,fs)

    R_XY = np.fft.ifft(S_XY)/delT
    sub = 0
    if N%2:
        N+=1
        sub =1

    R_XY = np.concatenate((R_XY[(N/2):],R_XY[0:N/2]))
    times = timeVec(N,fs)
    timeShift = np.concatenate((-1*times[1:(N/2)+1-sub][::-1],times[0:(N/2)]))

    return R_XY, timeShift

def crossCorrSpectrogram(x_time,y_time,fs,sliceLength,sync=0,overlap=0,color="jet",norm=True,pad=True):
    N = len(x_time)
    Nslices = int(N / sliceLength)
    T = Nslices * sliceLength / float(fs)

    _, timeShift, R_XY, C_XY = crossCorrArray(x_time,y_time,fs,sliceLength,pad,sync,overlap)

    TD = np.zeros(Nslices)

    for i in range(Nslices):
        TD[i] = timeShift[np.argmax(C_XY[i,:])]
    ext = [0, T, timeShift[0], timeShift[-1]]

    if norm:
        plt.imshow(np.abs(C_XY).T, aspect="auto", origin="lower", cmap=color, extent=ext)
    else:
        plt.imshow(abs(R_XY).T, aspect="auto", origin="lower", cmap=color, extent=ext)

    return TD

def crossCorrArray(x_time,y_time, fs, sliceLength,pad=True, sync=0,overlap=0):
    overlap = np.abs(overlap)
    if overlap >= 1.0:
        sys.exit("overlap >= 1")

    N = len(x_time)
    m = int((N - int(overlap * sliceLength)) / (sliceLength * (1 - overlap)))
    if pad:
        factor = 2
    else:
        factor = 1
    R_XY = np.zeros((m,factor*(sliceLength-1)), dtype=complex)
    C_XY = np.zeros((m,factor*(sliceLength-1)), dtype=complex)

    for i in range(m):
        n = i * (int(sliceLength*(1-overlap))+sync)
        sliceEnd = int(n + sliceLength - 1)
        x = x_time[n:sliceEnd]
        y = y_time[n:sliceEnd]

        if pad:
            x = np.append(x,np.zeros(len(x)))
            y = np.append(y,np.zeros(len(y)))
        x_RMS = rms(x,show=False)[0]
        y_RMS = rms(y,show=False)[0]
        R_XY[i,], timeShift = crossCorr(x,y, fs)
        C_XY[i,] = R_XY[i,]/(x_RMS*y_RMS)

    #### R_XY Avg
    R_XYavg = np.mean(R_XY, axis=0)

    return R_XYavg, timeShift, R_XY, C_XY

def coherence(x_time,y_time, fs, sliceLength, sync=0,overlap=0,winType="uniform"):
    G_XY,freq,_,_ = crossSpectroArray(x_time=x_time,y_time=y_time,fs=fs,sliceLength=sliceLength,sync=sync,overlap=overlap,winType=winType)
    G_XX,_,_,_ = spectroArray(x_time=x_time,fs=fs,sliceLength=sliceLength,sync=sync,overlap=overlap,winType=winType)
    G_YY,_,_,_ = spectroArray(x_time=y_time,fs=fs,sliceLength=sliceLength,sync=sync,overlap=overlap,winType=winType)

    gammaSqrd = (np.conj(G_XY)*G_XY)/(G_XX*G_YY)

    return gammaSqrd, freq

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

def octFilter(fc,fs):
    if fc/fs >0.32:
        sys.exit("Center frequency of third-octave filter too high for fs")
    if fc/fs <0.0013:
        sys.exit("Center frequency of third-octave filter too low for fs")

    N=3
    if fc/fs > 0.15:
        N=4
    alpha = 10**(0.15)
    f1_filt = fc/alpha
    f2_filt = fc*alpha
    print N
    print f1_filt
    print f2_filt
    print fs
    bb,aa = butterworth_BP(N,f1_filt,f2_filt,fs)

    return bb,aa

def butterworth_BP(N,f1_actual,f2_actual,fs):
    if f2_actual<f1_actual:
        sys.exit("ensure f1 < f2")
    if f2_actual> 0.99*fs/2:
        sys.exit("sample rate too low")

    # warp frequencies to account for bilinear transformation
    f1 = fs*tan(pi*f1_actual/fs)/pi
    f2 = fs * tan(pi * f2_actual / fs) / pi

    # calc geometric-mean center frequency
    fctr = np.sqrt(f1 * f2)

    # Angular location of poles
    theta = pi / N
    angs = np.arange((pi + theta) / 2,(3 * pi / 2),theta)

    w0 = 2*pi*fctr
    w02 = w0**2
    w_crit = 2*pi*(f2-f1)

    roots = w_crit*exp(1j*angs)

    dena = np.poly(roots).real
    mplus1 = len(dena)
    numa = dena[-1]

    nums = np.zeros(mplus1)
    nums[0] = numa

    dens = np.zeros(2*mplus1-1)
    tempd = dens

    m1 = mplus1
    m2 = m1
    m1 = m1-1
    ###
    rootv0 = np.array([-1j*w0,1j*w0])
    rootv = rootv0

    dens[m1] = dena[mplus1-1]
    print dens
    for i in range(mplus1-1):
        m1 = m1 - 1
        m2 = m2 + 1
        basic_coef = np.poly(rootv)

        tempd[m1:m2]=basic_coef
        dens = dens +dena[m1]*tempd
        rootv = np.concatenate((rootv,rootv0))


    bb,aa = bilinearXform(nums,dens,fs)
    return bb,aa