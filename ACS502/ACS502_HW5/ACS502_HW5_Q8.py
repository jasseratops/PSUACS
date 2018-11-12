import numpy as np
import sys

def main(args):
    alphaBar = 1.1E-3
    SPL33 = 130.

    r1 = 33.
    r2 = 800.

    rho0 = 1.21
    c0 = 343.
    p_ref = 20.E-6

    SPL800 = SPL33 + (20*np.log10(r1/r2))

    p_rms800 = p_ref*(10**(SPL800/20.))

    I800 = (p_rms800**2)/(rho0*c0)

    SPL800ABS = SPL800 - alphaBar*(r2-r1)

    p_rms800ABS = p_ref*(10**(SPL800ABS/20.))
    I800ABS = (p_rms800ABS**2)/(rho0*c0)

    print "SPL800: " + str(SPL800) + " [dB]"
    print "I800: " + str(I800) + " W/m"
    print "SPL800 ABS: " + str(SPL800ABS) + " [dB]"
    print "I800 ABS: " + str(I800ABS) + " W/m"

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))