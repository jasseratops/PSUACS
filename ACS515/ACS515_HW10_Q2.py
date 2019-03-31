# PSUACS
# ACS515_HW10_Q2
# Jasser Alshehri
# Starkey Hearing Technologies
# 3/30/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, cos
from scipy.special import spherical_jn as jn, spherical_yn as yn, lpn

def main(args):
    deg = 0.01
    degRes = deg / 360.
    theta = pi * np.arange(0, 2, degRes)

    ka = [2.,4.,8.]
    kr = 1000.

    upper = 100   # Sets the upper limit of the summation


    plt.figure()
    for i in ka:
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, np.abs(velPotSum(i,kr,upper,theta)),label="ka: "+str(i))
        ax.grid(True)
    plt.legend()
    plt.show()

    return 0

def velPotSum(ka,kr,upper,theta):
    # Calculates the Sum of the Velocity Potential iterations for the complate solution
    velPot = np.zeros_like(theta,dtype=complex)
    for i in range(upper / 2):
        n = 2 * i + 1
        velPot += velPotIter(ka, kr, n, theta)

    return velPot

def velPotIter(ka,kr,n,theta):
    # Calculates single iteration of the Velocity Potential
    result = np.zeros_like(theta,dtype=complex)
    u0 = 1.
    a = 1.
    k = ka/a
    An = (u0/(k*hankDer(n,ka)))*(((2.*n)+1.)/(n+1.))*(lpn(n-1,0)[0][-1])
    hn2 = hank(n,kr)

    for i in range(len(theta)):
        result[i] = An*hn2*(lpn(n, cos(theta[i]))[0][-1])

    return result

def hank(n,z):
    # Calculates the Hankel function
    result = jn(n,z)-1j*yn(n,z)
    return result

def hankDer(n,z):
    #Caclulates the first derivative of the hankel function with respect to z
    result = -1*hank(n+1,z)+(n/z)*hank(n,z)
    return result

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))