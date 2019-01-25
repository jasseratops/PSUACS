# PSUACS
# ACS515_HW2_Q4
# Jasser Alshehri
# Starkey Hearing Technologies
# 1/24/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    k_x = 6*pi
    k_y = 0
    i = -1j
    k_z = i*4*pi

    k = np.sqrt((k_x**2)+(k_y**2)+(k_z**2))

    print k

    print k/k_x
    print k/(2*pi)
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))