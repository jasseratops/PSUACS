# JAscripts
# ExpSeries
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/3/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):

    A = 1
    alpha = -0.5
    n = 50
    k = np.arange(0,n+1)
    x = A*(alpha**(k))

    print str(x)

    plt.figure()
    plt.plot(k,x)
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))