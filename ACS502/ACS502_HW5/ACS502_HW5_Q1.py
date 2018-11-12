from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(args):
    L = 10.
    y_th = 0.005
    y_m = 5
    c0 = 343.

    m = (2./L)*np.log(y_m/y_th)
    print m
    fc = (m*c0)/(4*np.pi)
    print fc

    rho0 = 1.21
    S0 = np.pi*(y_th**2)
    u0 = 3.
    f = 2000.

    #f = np.linspace(40.,6000.,2**12)

    W_exp = 0.5*rho0*c0*S0*(u0**2)*np.sqrt(1-((fc/f)**2))
    print W_exp

    omega  = 2*np.pi*f
    k0 = omega/c0
    r0 = L

    W_con = 0.5*rho0*c0*S0*(u0**2)*((k0*r0)**2)/(1+((k0*r0)**2))
    '''
    plt.figure()
    plt.plot(f,W_exp)
    plt.plot(f,W_con)
    plt.show()
    '''
    print W_con

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))