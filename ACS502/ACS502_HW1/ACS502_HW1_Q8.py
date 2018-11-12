# PSUACS
# ACS502_HW1_Q8
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/29/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    c = 343

    p0 = 10E-8
    R = 8.314
    d = 2E-10
    Na= 6.022E23

    T = 100

    mfp = (R*T)/(np.sqrt(2)*pi*(d**2)*p0*(Na))

    print mfp

    f = c/mfp
    print f

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))