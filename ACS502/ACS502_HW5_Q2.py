import numpy as np
from numpy import log10, exp
import sys

def main(args):

    r_s = 50.
    r_b = 500.
    SPL_SR = 137.
    SPL_BR = 100.

    rho1 = 1026.
    c1 = 1500.
    z1 = rho1*c1

    rho2 = 1.21
    c2 = 343.
    z2 = rho2*c2

    R_S = abs((z2-z1)/(z2+z1))

    p_ref = 1.E-6

    p_SR = p_ref * (10 ** (SPL_SR / 20.))
    p_BR = p_ref * (10 ** (SPL_BR / 20.))

    p_0 = p_SR*(r_s**2)/abs(R_S)
    print R_S
    print (p_BR/p_0)*(r_b**2)

    print 20*log10(p_0/p_ref)

    p_BR0 = p_BR*r_b
    p_iB = p_0/r_b

    R_B = p_BR0/p_iB

    print "SR: " + str(p_SR)
    print "BR: " + str(p_BR)
    print "0:  " + str(p_0)
    print "R_B: " + str(R_B)

    pH = 8.

    c0 = 1500.      # m/s
    S = 35.#/1000.   # ppt

    t = 13.     # C

    f = 50.     # kHz

    D = 275.        # m
    c = 1412. + 3.21*t + 1.19*S + 0.0167*D

    print c0
    print c

    alphaBar = OCEAN_ABSORB(f, c0, S, pH, t)/1000.
    alpha = alphaBar/8.686
    print alphaBar
    print alpha

    p_0 = (p_SR*(r_s**2))/(R_S*exp(-2*alpha*r_s))

    R_B = (p_BR*(r_b**2))/(p_0*exp(-2*alpha*r_b))


    print p_0
    print R_B
    return 0

def OCEAN_ABSORB(f,c,S,pH,t):
    f1 = 2.8*((S/35.)**0.5)*(10.**(4.-1245./(t+273.)))
    f2 = 8.17*(10.**(8.-1990./(t+273.)))/(1+0.0018*(S-35.))

    A1 = (8.86/c)*10**((0.78*pH) - 5)
    A2 = ((21.44*S)/c)*(1+(0.025*t))
    A3 = (4.937E-4) - ((2.59E-5)*t) + ((9.11E-7)*(t**2)) - ((1.50E-8)*(t**3))

    P1 = P2 = P3 = 1.


    alphaBar = ((A1 * P1 * f1 * (f ** 2)) / ((f ** 2) + (f1 ** 2))) + (
    (A2 * P2 * f2 * (f ** 2)) / ((f ** 2) + (f2 ** 2))) + (A3 * P3 * (f ** 2))

    return alphaBar

if __name__ == "__main__":
    sys.exit(main(sys.argv))