import numpy as np
from numpy import pi,sqrt

def main(args):
    m = 17.2E-3
    d = 1.55E-2
    L = 14.96E-2
    fT = 4192
    fL = 6952

    vol = pi*((d/2)**2)*L
    rho = m/vol

    Y = 4*rho*(L**2)*(fT**2)
    G = 4*rho*(L**2)*(fL**2)

    v = (Y/(2*G))-1

    print "Y: ", Y
    print "G: ", G
    print "v: ", v

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))