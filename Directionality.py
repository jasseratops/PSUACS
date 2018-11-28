# PSUACS
# Directionality
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/20/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import scipy.signal as sig

def main(args):
    fs = 756.E3
    T = 0.1
    N = int(fs*T)

    c = 343.
    portSpacing = 6.E-3
    portDelay = portSpacing/c



    eDelay = 4.E-6

    dur = 0.1
    #stimulus = sig.chirp(t=)
    fMicInput = np.zero

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))