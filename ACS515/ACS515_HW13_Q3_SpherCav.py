# PSUACS
# ACS515_HW13_Q3
# Jasser Alshehri
# Starkey Hearing Technologies
# 4/23/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp

def main(args):
    c_sw = 1500.
    c_air = 343.
    rho_sw = 1026.
    f_100_sw = 1000.
    l = 1.

    a = (l*c_sw)/(2*f_100_sw)

    V = (pi*(a**2))
    m3_to_gal = 264.172
    V_gal = V*m3_to_gal
    print V_gal
    m_sw = V*rho_sw/1000.       # weight in metric tons

    print m_sw
    z_11 = 2.082
    f_100_air = (z_11*c_air)/(2*a)
    print f_100_air

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))