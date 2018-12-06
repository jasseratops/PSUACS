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

    D = 0.707
    f = np.arccos(D)*c/(pi*d)
    print f

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))