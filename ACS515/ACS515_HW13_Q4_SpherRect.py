import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp

def main(args):
    z_11 = 2.08
    rigCube = 1./np.cbrt((4.*pi/3.))
    rigSphe = z_11/pi
    print "Rigid:"
    print "cube: " + str(rigCube)
    print "sphere: " + str(rigSphe)
    print "perc less: " + str(perc(rigCube,rigSphe))
    print "-"*10
    print "Release:"
    relSphe = 1.
    relCube = 1./(np.cbrt((4.*pi/3.)))
    print "cube: " + str(relCube)
    print "sphere: " + str(relSphe)
    print "perc less: " + str(perc(relCube,relSphe))

    return 0

def perc(a,b):
    dif = np.abs(a-b)
    per = dif/np.max([a,b])
    return per

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))