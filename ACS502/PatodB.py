# PSUACS
# PatodB
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/22/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    Pa = 2./(np.sqrt(2))
    ref = 20.0E-6

    dB = 20*np.log10(Pa/ref)

    print dB
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))