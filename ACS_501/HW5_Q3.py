# ACS_501
# HW5_Q3
# J. Alshehri
# 10/29/17

import math


def main(args):
    L = 0.2         # m
    mBar = 0.04     # kg
    m1 = 0.027      # kg
    m2 = 0.054      # kg
    omega = 2.0*math.pi*8989.0
    c = 5050.0

    k = omega/c


    x = m1/mBar
    y = m2/mBar

    A = /math.tan(k*L)
    B = (-(k*L)*y)/(1-((k*L)**2)*x)
    print A
    print B

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
