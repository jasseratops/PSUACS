# PSUACS
# HA_Mic_ka
# Jasser Alshehri
# Starkey Hearing Technologies
# 5/8/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    c = 343.
    frq = np.array([100.,10.E3])
    k = frq*2*pi/c
    a = 3.55E-3          # largest dimension of the 5100t

    ka = k*a

    print ka

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))