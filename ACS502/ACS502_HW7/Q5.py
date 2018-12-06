import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt
from scipy.special import jv

def main(args):
    c = 1480.
    rho= 1000.
    a = 10.E-3
    f = 1.48E6
    omega = 2*pi*f

    k = omega / c

    lam = c/f

    print lam

    R0 = (k*(a**2)/2)
    print R0

    m = np.sqrt(400.*pi)
    print "m: " + str(m)
    ms = range(35)[::2][1:]


    deg = 0.1
    degRes = deg/180.
    theta = np.pi * np.arange(0.01,0.5,degRes)

    D = 2*jv(1,(k*a*sin(theta)))/(k*a*sin(theta))

    zc = np.where(np.diff(np.signbit(D)))[0]
    print len(zc)

    p_ax_rms = 5.01
    u0 = (np.sqrt(2)/2.)*(1/(rho*c))*(1./sin(1./8.))*p_ax_rms
    print "u0: " + str(u0)

    r = 49.5E-3

    pNew = ((rho*(a**2))/(2*r))*omega*u0
    pNewRMS = pNew/np.sqrt(2)
    SPL_new = 20*np.log10(pNewRMS/(1.E-6))
    print pNewRMS
    print SPL_new

    plt.figure()
    plt.plot(theta,D)
    for i in zc:
        plt.axvline(theta[i])
    plt.title("Q5: zero crossings")
    plt.ylabel("2J_1(ka)/(ka)")
    plt.xlabel("ka")

    plt.show()


    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))