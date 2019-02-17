# PSUACS
# ACS515_HW5_Q1
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/16/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    ka = 0.1
    rat = np.sqrt((3*(4+(ka**4)))/((ka**2)*(1+(ka**2))))

    print rat

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))