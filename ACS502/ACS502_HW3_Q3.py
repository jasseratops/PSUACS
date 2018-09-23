# PSUACS
# ACS502_HW3_Q3
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/22/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
from numpy import radians as rad
from numpy import degrees as deg


def main(args):
    trAng5 = rad(5.0)
    trAng6 = rad(6.0)

    c_f_water = 1481.0
    c_air = 343

    c1 = c_f_water
    c2 = c_air

    ang5 = np.arcsin((c1 / c2) * trAng5)
    ang6 = np.arcsin((c1 / c2) * trAng6)

    print deg(ang5)
    print deg(ang6)

    d = 1.0/(tan(ang6)-tan(ang5))

    print d

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))