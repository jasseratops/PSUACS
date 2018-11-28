# PSUACS
# PolarTest
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/20/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    deg = 10.
    degRes = deg/360.
    theta = 2 * np.pi * np.arange(0,2,degRes)

    a = 1.
    r = a*(1+sin(theta))-1*sin(pi+theta)

    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, r)
    ax.set_rmax(2)
    ax.set_rticks([0.25, 0.5,0.75, 1])  # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)

    ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))