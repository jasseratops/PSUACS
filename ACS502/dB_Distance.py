# PSUACS
# dB_Distance
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/25/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):

    dB1 = 44.0
    r1 = 450.0
    r2 = 1000.0

    dB2 = dB1 + 20*np.log10(r1/r2)

    print dB2

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))