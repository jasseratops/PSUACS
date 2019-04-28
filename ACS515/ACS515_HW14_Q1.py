import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp

c = 1500.
Lx = 0.220
Ly = 0.175

def main(args):
    print "frq11: " + str(cutOnFrq(1, 1))
    print "frq12: " + str(cutOnFrq(1, 2))
    print "frq21: " + str(cutOnFrq(2, 1))
    print "frq22: " + str(cutOnFrq(2, 2))
    return 0

def cutOnFrq(m,n):
    frq = (c/2.)*np.sqrt((m/Lx)**2 + (n/Ly)**2)
    return frq

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))