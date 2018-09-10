# PSUACS
# dBtoPa
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/25/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    dB = 185.0
    ref = 1.0E-6
    Pa = (10.0**(dB/20.0))*ref

    print Pa

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))