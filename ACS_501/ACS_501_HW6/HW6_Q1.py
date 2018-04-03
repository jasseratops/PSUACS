# JAscripts
# HW6_Q1
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/9/2017


from numpy import pi, sin, cos, tan, exp
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    cC = 3700
    cS = 5050
    rhoC = 8900
    rhoS = 7700
    r1 = 0.1
    r2 = 0.141
    L = 1

    S1 = 2*pi*r1
    S2 = 2*pi*r2

    ZmS = rhoS*cS*S1
    ZmC = rhoC*cC*S2

    k = np.append([0.1], np.linspace(1,10,1000, endpoint=True))
    print k

    Zm0 = ZmS*(((-ZmC/ZmS)+1j*tan(k*L))/(1+((1j*ZmC*tan(k*L))/ZmS)))

    plt.figure()
    plt.plot(k,abs(Zm0), label= "Magnitude")
    plt.plot(k,Zm0.real, label= "Real")
    plt.plot(k,Zm0.imag, label= "Imaginary")
    plt.xlabel("k")
    plt.ylabel("Impedance")
    plt.legend()
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))