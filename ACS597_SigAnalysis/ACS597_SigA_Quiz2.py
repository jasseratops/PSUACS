# PSUACS
# ACS597_SigA_Quiz2
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/24/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import soundfile as sf
import sigA


def main(args):
    part1()
    return 0

def part1():
    data, fs = sf.read("N96_2017_3BB.wav")
    N = len(data)
    times = sigA.timeVec(N,fs)
    t = 11.
    frst11 = data[0:int(fs*t)]
    scnd11 = data[int(fs*t)]

    #msAvgd = sigA.

    GxxAVG

    plt.figure()
    plt.plot(times,data)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    plt.title("")
    plt.show()
    return 0
if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))