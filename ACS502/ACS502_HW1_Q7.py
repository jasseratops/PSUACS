# PSUACS
# ACS502_HW1_Q7
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/30/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    c = 331.65
    N = 6.022E23
    Neu = 1.0E6

    volL = 22.413996
    VOLeu = Neu*(volL/N)
    VOLeuCM = VOLeu*1E-3

    print VOLeu
    print VOLeuCM
    LENeu = np.cbrt(VOLeuCM)
    print "LENeu: " + str(LENeu)

    f = c/(LENeu*10)

    print f

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))