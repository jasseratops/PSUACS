import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt


def main(args):
    kh = np.linspace(0.1,30,1024)
    rat = (sin(2*kh)/(2*kh))+1

    plt.figure()
    plt.plot(kh,rat)
    plt.axhline(1,linestyle="--")
    for i in range(int(max(kh))):
        plt.axvline(i*pi/2,linestyle=":")

    plt.xlim(kh[0],kh[-1])
    plt.xlabel("kh")
    plt.ylabel(r"W/$W_{free}$")
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))