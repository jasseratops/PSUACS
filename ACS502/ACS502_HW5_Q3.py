import numpy as np
import matplotlib.pyplot as plt
import sys

def main(args):
    f = 10.E3
    omega = 2 * np.pi * f
    mu = 0.96
    rho0 = 950.
    c0 = 1540

    alpha1 = ((mu*(omega**2))/(2*rho0*(c0**2)))*(4./3.)
    alpha2 = ((mu*((omega*2)**2))/(2*rho0*(c0**2)))*(4./3.)

    print alpha1
    print alpha2

    x = 14./(8.686*(alpha2-alpha1))
    print x

    return 0

main(sys.argv)