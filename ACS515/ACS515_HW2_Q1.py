# PSUACS
# ACS515_HW2
# Jasser Alshehri
# Starkey Hearing Technologies
# 1/22/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    AA= 1.
    ff = 100.
    omega= 2.*pi*ff
    tt= 0.5
    i = -1j
    cc = 343.
    xx = np.linspace(0.,10.,1024)
    kk = omega/cc
    p1= AA*cos(omega*tt)*sin(kk*xx)
    p2 = (AA/2)*(sin((omega*tt)+(kk*xx))-sin((omega*tt)-(kk*xx)))

    print p2

    plt.figure()
    plt.plot(xx,p1)
    plt.plot(xx,p2)
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))