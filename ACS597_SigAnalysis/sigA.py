# PSUACS
# sigA
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/1/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp

import sounddevice as sd

def param(N,fs,show=True):
    delT = 1/float(fs)
    delF = float(fs)/float(N)
    T = float(N)/float(fs)
    if show:
        print "N: " + str(N)
        print u"\N{GREEK CAPITAL LETTER DELTA}" + "t: " + str(delT)`
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



def linSpec(x_time,fs):
    N = len(x_time)
    delT, _, _ = param(N, fs, False)

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


def PnN_LinSpec(x_time,fs):
    N = len(x_time)
    ind = (N/2)+1               # Rounds down if length of x_time is odd
    lsp = linSpec(x_time,fs)
    pos_lsp = lsp[:ind]
    neg_lsp = lsp[ind:]

    return pos_lsp, neg_lsp

def dsSpec(x_time,fs):
    N = len(x_time)
    _, delF, _ = param(N, fs, False)
    lsp = linSpec(x_time,fs)
    Sxx = (abs(lsp)**2)*delF

    return Sxx

def ssSpec(x_time,fs):
    Sxx = dsSpec(x_time,fs)
    N = len(Sxx)

    Gxx = Sxx[0:(N/2)+1]
    odd = bool(len(Sxx)%2)

    for i in range(len(Gxx)):
        if (i != 0) or (i==(len(Gxx)-1) and odd):
            Gxx[i] = (Gxx[i])*2

    return Gxx

def play(x_time,fs):
    normFactor = 0.8/max(x_time)
    sdArray = (x_time*normFactor)
    sd.play(sdArray,fs)