import numpy as np
from numpy import exp, sqrt
import sys

def main(args):
    I_Eth = 841.
    f = 3.9
    alphaBar_soft = .30 *f
    alphaBar_Bone = 8.70 *f
    alphaBar_Eth  = 0.0044 *(f**2)

    print (IAbs(I_Eth,4.,alphaBar_soft))
    print (IAbs(I_Eth, 4., alphaBar_Bone))
    I_init = backwards(I_Eth, 6., alphaBar_Eth)
    print (IAbs(I_init,4.,alphaBar_soft))
    print (IAbs(I_init, 4., alphaBar_Bone))
    return 0


def IAbs(I,dist,alphaBar):
    alpha = alphaBar/8.686
    I_Absorb = I*exp(-2*alpha*dist)
    return I_Absorb#/1.E4

def backwards(I,dist,alphaBar):
    alpha = alphaBar/8.686
    I_init = I*exp(2*alpha*dist)
    return I_init

if __name__ == "__main__":
    sys.exit(main(sys.argv))