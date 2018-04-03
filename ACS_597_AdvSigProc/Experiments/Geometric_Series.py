# JAscripts
# Geometric_Series
# Jasser Alshehri
# Starkey Hearing Technologies
# 1/30/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):

    alpha = 0.50    # Alpha of Geom. Series.
    N = 1000        # should be infinity
    S = 0.00        # initialize SUM

    for k in range(N):
        S += alpha**k   # Summation
        print S

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))