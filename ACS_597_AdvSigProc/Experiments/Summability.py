# JAscripts
# Summability
# Jasser Alshehri
# Starkey Hearing Technologies
# 3/7/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    N = 100
    n = np.arange(-N,N)
    w = pi/5.0
    sum = 0
    sumSq = 0
    x = (sin(w*n))/(pi*n)
    #x[100]=1

    xAbs = abs(x)
    xAbsSq = (abs(x))**2

    for i in range(len(n)):
        print "n: " + str(i-N)
        print "x[n]: " + str(x[i])
        print "|x[n]|: " + str(xAbs[i])
        print "|x[n]|^2: " + str(xAbsSq[i])
        sum += xAbs[i]
        sumSq += xAbsSq[i]

        print "Sum: " + str(sum)
        print "SumSq: " +str(sumSq)

    plt.figure()
    plt.plot(n,x,label="x[n]")
    plt.plot(n,xAbs,label="|x[n]|")
    plt.plot(n,xAbsSq,label="|x[n]|^2")
    plt.legend()
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))