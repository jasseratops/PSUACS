import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt

def main(args):
    rho0 = 1.21
    c0 = 343.
    kr = np.linspace(0.1,10.,1000)
    Z0 = rho0*c0

    p_u = np.abs((rho0*c0)/(1.-(1j/kr)))

    plt.figure()
    plt.plot(kr,p_u)
    plt.axhline(Z0)
    plt.title("Z vs kr")
    plt.xlabel("kr")
    plt.ylabel("|Z|")
    plt.xlim(kr[0],kr[-1])
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))