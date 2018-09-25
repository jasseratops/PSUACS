# PSUACS
# ACS502_HW3_Q6
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/25/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    c0 = 343.
    T0 = 273.15
    dTdh = (10-23)/(5-0.7)

    dcdh_temp = (c0/(2*T0))*(dTdh)

    print dcdh_temp

    u = 3.
    A = 0.143
    dcdh_wind = u*A

    print dcdh_wind
    hs = 10.
    hr = 2.
    beta = np.radians(25.)

    x = np.sqrt((2*c0)/((cos(beta)*dcdh_wind)-dcdh_temp))*(np.sqrt(hs)+np.sqrt(hr))

    print x

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))