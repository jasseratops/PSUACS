import numpy as np
from numpy import pi,sqrt

T = 25000
a = 0.25
rhoS = 1.0


def main(args):
    jmn = [2.40,3.83,5.14,5.52]
    for i in jmn:
        freqCalc(i)

def freqCalc(jmn):
    freq = (jmn/(2*pi*a))*sqrt(T/rhoS)
    print freq

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))