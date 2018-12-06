import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt
from scipy.special import jv

def main(args):
    x = np.linspace(0.,14.,1024*4)
    Bessel = jv(1,x)
    curve = 2*Bessel/x
    print Bessel
    a = 6.35E-3
    c = 343.


    plt.figure()
    plt.plot(x,curve)
    for i in range(len(curve)):
        if round(curve[i],3) == 0.707:
            plt.axvline(x[i])
            ka = (2./0.707)*Bessel[i]
            print "hi"

    k = ka/a
    omega = k*c
    f = omega/(2*pi)
    print f
    plt.axhline(0)


    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))