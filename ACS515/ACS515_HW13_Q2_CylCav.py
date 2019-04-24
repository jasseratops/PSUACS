# PSUACS
# ACS515_HW13_Q2
# Jasser Alshehri
# Starkey Hearing Technologies
# 4/23/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import scipy.special as sp

L = 2.
a = 3.
c = 1481.

def main(args):
    m = 5
    n = 5
    x = 5

    alpha = np.zeros((m,n))
    for i in range(m):
        alpha[i][:] = sp.jnp_zeros(i,5)
        if i == 0:
            alpha[i][:] = np.roll(alpha[i][:],1)
            alpha[i][0] = 0

    print alpha

    frqs = np.zeros((5,5,5))
    print "(m,n,l)"
    print "-"*10

    for i_m in range(m):
        for i_n in range(n):
            for i_x in range(x):
                i_l = i_x*2 + 1

                frqs[i_m, i_n, i_x] = nat_frq(alpha[i_m,i_n],i_l)

    B = np.sort(frqs,axis=None)

    print "Minima"
    print "-"*10

    for i_m in range(m):
        for i_n in range(n):
            for i_x in range(x):
                if frqs[i_m, i_n, i_x] in B[:5]:
                    i_l = i_x * 2 + 1
                    print "(" + str(i_m) + ","+ str(i_n+1) + "," + str(i_l) + "): " + str(frqs[i_m, i_n, i_x])

    return 0

def nat_frq(alpha,l,):
    frq = (c/2.)*np.sqrt(((alpha/(a*pi))**2)+((l/(2*L))**2))
    return frq

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))