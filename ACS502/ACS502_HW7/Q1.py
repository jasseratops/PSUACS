import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt


def main(args):
    c = 343.
    rho0=1.21
    d = 0.16
    print c/(pi*d)
    print 10**(-3./20.)
    f_null = c/(2*d)
    omega= 2*pi*f_null
    k = omega/c


    D = 0.707
    f = np.arccos(D)*c/(pi*d)
    print f

    print "f_null: " + str(f_null)
    print k*d
    theta = np.degrees(np.arcsin(pi/(k*d)))
    print theta

    f_null2 = c/(d)
    print f_null2

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))