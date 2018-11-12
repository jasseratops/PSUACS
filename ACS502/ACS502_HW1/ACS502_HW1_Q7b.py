# PSUACS
# ACS502_HW1_Q7b
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/30/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    c = 1486.0
    rhoW = 998.0
    MmW = 18.015E-3
    N = 6.022E23

    invMolDens = MmW/(N*rhoW)

    VOLeu = (1.0E6)*invMolDens

    LENeu = np.cbrt(VOLeu)

    f = c/(10.0*LENeu)

    print "invMolDens: " + str(invMolDens)
    print "VOLeu: " + str(VOLeu)
    print "LENeu: " + str(LENeu)
    print "f: " + str(f)

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))