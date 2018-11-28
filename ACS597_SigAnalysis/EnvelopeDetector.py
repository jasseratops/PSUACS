# PSUACS
# EnvelopeDetector
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/23/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import sigA
import scipy.signal as sig

def main(args):
    fs = 48000.
    T = 5.
    N = int(T*fs)

    dur = 1.
    durInd = int(dur*fs)

    stim = np.zeros(N)

    for i in range(N):
        if i < durInd:
            stim[i] = sin

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))