# PSUACS
# dBtoPa
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/25/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    dB = 117.
    ref = 20.0E-6
    p_rms = (10.0**(dB/20.0))*ref
    rmsFactor = np.sqrt(2.)
    p_pk = p_rms*rmsFactor
    print "Prms: " + str(p_rms)
    print "Ppk:  " + str(p_pk)

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))