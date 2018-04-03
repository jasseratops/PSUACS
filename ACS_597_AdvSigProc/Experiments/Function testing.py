# JAscripts
# Function testing
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/6/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    n = 50
    u= np.zeros(n)
    h_n= np.zeros(n)
    x = -0.5*n
    for i in range(len(h_n)):
        x +=i
        if x<0:
            u[i]= 0
        else:
            u[i]= 1
        h[i] = ((1/3)^x)*u[i]




    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))