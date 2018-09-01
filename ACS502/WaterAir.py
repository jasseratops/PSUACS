# PSUACS
# WaterAir
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/25/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    freq = 0.5


    cAir = 343.0
    cWater = 1500.0

    wlAir = cAir/freq
    wlWater = cWater/freq

    kAir = 2*pi/wlAir
    kWater = 2*pi/wlWater

    print "Wavelength in Air: " + str(wlAir) + "[m]"
    print "Wave Number in Air: " +str(kAir) + "[rad/m]"
    print 10*"-"

    print "Wavelength in Water: " + str(wlWater) + "[m]"
    print "Wave Number in Water: " + str(kWater) + "[rad/m]"


    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))