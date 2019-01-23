# PSUACS
# ACS515_HW2_Q2
# Jasser Alshehri
# Starkey Hearing Technologies
# 1/22/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    m2 = 1.
    m3 = 1.

    t2 = 2.55E-3
    t3 = 1.47E-3

    c_x = m2/t2
    c_y = m3/t3

    print c_x
    print c_y

    c = np.sqrt((c_x**2)+(c_y**2))
    thet = np.degrees(np.arctan(c_x/c_y))

    print c
    print thet

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))