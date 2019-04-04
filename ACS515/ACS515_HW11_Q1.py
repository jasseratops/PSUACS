# PSUACS
# ACS515_HW11_Q1
# Jasser Alshehri
# Starkey Hearing Technologies
# 4/3/2019

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    a = 2.5E-2
    r = 8.
    f = 100.
    c = 343.
    SPL = 95.
    p_ref = 20.E-6
    rho = 1.21

    omega = 2*pi*f
    k = omega/c

    ka = k*a
    kr = k*r

    p_pk = p_ref*10**(SPL/20.)

    debug('kr')
    debug('ka')



    return 0

def debug(expression):
    frame = sys._getframe(1)
    result = (expression, ': ', repr(eval(expression, frame.f_globals, frame.f_locals)))
    for i in result:
        print(i,end ="")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))