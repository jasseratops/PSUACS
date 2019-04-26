import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt
import scipy.special as sp

c = 343.
def main(args):
    f = 4000.
    a = 5.E-2
    aP_21 = alphaP_mn(2,1)
    aP_02 = alphaP_mn(0,2)

    f_21 = cutOn(aP_21,a)
    f_02 = cutOn(aP_02,a)

    print "Cut-on, 21: " + str(f_21)
    print "Cut-on, 02: " + str(f_02)

    print "Phase Speed: " + str(phaseSpeed(f_21,f))
    print "Group Speed: " + str(groupSpeed(f_21,f))

    return 0

def cutOn(alphaP,radius):
    f = alphaP*c/(2*pi*radius)
    return f

def alphaP_mn(m,n):
    alpha = sp.jnp_zeros(m,n)
    if m == 0:
        alpha = np.roll(alpha,1)
        alpha[0] = 0

    return alpha[n-1]

def phaseSpeed(f_mn,f):
    c_p = c/np.sqrt(1.-(f_mn/f))
    return c_p

def groupSpeed(f_mn,f):
    c_g = c*np.sqrt(1.-(f_mn/f))
    return c_g

if __name__ == "__main__":
    sys.exit(main(sys.argv))