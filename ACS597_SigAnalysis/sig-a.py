# PSUACS
# sig-a
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/1/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp

def dispParam(N,fs):
    delT = 1/float(fs)
    delF = float(fs)/float(N)
    T = N*fs

    print "N: " + str(N)
    print u"\N{GREEK CAPITAL LETTER DELTA}" + "t: " + str(delT)
    print u"\N{GREEK CAPITAL LETTER DELTA}" + "f: " + str(delF)
    print "fs: " + str(fs)
    print "T: " + str(T)
    print 10 * "-"

    return delT, delF, T


def timeVec(N, fs):
    delT = 1.0 / float(fs)
    t = np.arange(0, N) * delT

    return t


def freqVec(N, fs):
    delF = float(fs) / float(N)
    f = np.arange(0, N) * delF

    return f

def check(x_time,fs):
    length = len(x_time)
    if length < 4:
        print "Array size: " + str(length)
        print "Array too small"



def linSpec(x_time,fs):
    delT = 1.0/float(fs)

    return np.fft.fft(x_time)*(1/delT)

def timeSer(lsp, fs):
    delF = float(fs)/float(N)
    x_time  = np.fft.ifft(lsp)*delF
    return x_time

def rms(x_time):
    m_x_time = np.mean(x_time.real)
    ms_x_time = np.mean((x_time.real)**2)
    rms_x_time = np.sqrt(ms_x_time)

    return rms_x_time, ms_x_time, m_x_time


def PnN_LinSpec(x_time,fs):
    N = len(x_time)
    ind = (N/2)+1               # Rounds down if length of x_time is odd
    lsp = linSpec(x_time,fs)
    pos_lsp = lsp[:ind]
    neg_lsp = lsp[ind:]

    return pos_lsp, neg_lsp
