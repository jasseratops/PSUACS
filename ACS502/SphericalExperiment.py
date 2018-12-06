import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt

def main(args):
    A = 10.
    kr = np.linspace(0.01,20.,1024)
    f = 100.
    omega = 2*pi*f
    c = 343.
    k = omega/c
    r = kr/k
    t = 0.
    rho0 = 1500.

    p = (A/r)*exp(1j*((omega*t) - kr))
    u = (1.-(1j/kr))*(1/(rho0*c))*(p)
    I = u*p
    W = I*(4*pi*(r**2))
    Z = p/u
    ang = np.arctan(Z)#*(180./pi)
    #ang = np.degrees(ang)
    print ang
    print p
    plt.figure()
    plt.subplot(311)
    plt.plot(kr,np.abs(p))
    plt.plot(kr,np.real(p))
    for i in range(len(p)-1):
        arr = np.real(p)
        if (arr[i] > arr[i-1]) and (arr[i] > arr[i+1]):
            plt.axvline(kr[i])
        elif (arr[i] < arr[i-1]) and (arr[i] < arr[i+1]):
            plt.axvline(kr[i])

    #plt.xlim(0,10)
    plt.ylim(-A,A)

    plt.title("p")
    plt.subplot(312)
    plt.plot(kr,np.abs(u))
    plt.plot(kr,np.real(u))
    for i in range(len(u)-1):
        arr = np.real(u)
        if (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]):
            plt.axvline(kr[i])
        elif (arr[i] < arr[i - 1]) and (arr[i] < arr[i + 1]):
            plt.axvline(kr[i])

    plt.title("u")
    #plt.xlim(0,10)
    plt.subplot(313)
    plt.plot(kr,ang)
    plt.xlim(kr[0],1)

    plt.figure()
    plt.subplot(311)
    plt.plot(kr,(I))
    plt.title("I")
    plt.subplot(312)
    plt.plot(kr,(W))
    plt.title("W")
    plt.subplot(313)
    plt.plot(kr,Z)
    plt.title("Z")
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))