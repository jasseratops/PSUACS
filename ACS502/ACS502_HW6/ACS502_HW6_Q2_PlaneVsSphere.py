# PSUACS
# ACS502_Q2_PlaneVsSphere
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/15/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    f = 7000.
    r = 3.E-3
    SPL = 117.
    rho = 1.21
    c = 343.
    pref = 20.E-6

    Z = rho*c
    omega = 2*pi*f
    k = omega/c

    prms = pref*10**(SPL/20)
    print prms
    urms = prms/Z
    print urms
    xirms = urms/omega
    print xirms

    ### sphere
    A = r*np.sqrt(2)*prms
    print "A: "+str(A)
    u = np.abs((1-1j/(k*r))*(1/Z)*(A/r))
    print "u: " + str(u)
    uSrms = u/np.sqrt(2)
    print "uSrms: " + str(uSrms)
    xiSrms = uSrms/omega
    print "xiSrms: " + str(xiSrms)

    ### 1.05 apart

    factor = 1.+0.05

    poly = [prms/(factor*A),-1,1j/k]
    poly = [1.-1.05,1.05j/k,0]

    print np.roots(poly)

    kr = np.linspace(0.1,10.,4096)
    r = kr/k

    u_plane = np.ones(len(r))*urms
    u_sphere = np.abs((1-(1j/(k*r)))*(1/Z)*(A/r)/np.sqrt(2))
    top = abs(u_sphere-(1.05*u_plane))
    bot = abs(u_sphere - (0.95 * u_plane))
    topR = r[np.argmin(top)]
    botR = r[np.argmin(bot)]
    print topR
    print botR

    plt.figure()
    plt.plot(r,u_plane,label="u_rms_plane")
    plt.plot(r,u_sphere,label="u_rms_sphere")

    plt.axhline(u_plane[0]*1.05,linestyle=":",color="black")
    plt.axhline(u_plane[0]*0.95,linestyle=":",color="black")

    plt.axvline(topR, linestyle="--",color="black")
    plt.axvline(botR, linestyle="--",color="black")
    plt.xlim(r[0],r[-1])
    plt.xlabel("r [m]")
    plt.ylabel("particle velocity [m/s]")
    plt.legend()

    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))