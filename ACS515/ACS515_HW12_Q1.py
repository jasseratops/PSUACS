import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt
from scipy.special import spherical_jn as jn, spherical_yn as yn, hankel2
from math import isnan, isinf

def main(args):
    n = np.arange(100)
    ka = np.geomspace(0.01,50,1024)

    F_bs = (4./(ka**2))*(np.abs(summingRel(n,ka)))**2

    plt.figure()
    plt.semilogx(ka,F_bs)
    plt.axvline(1,color="black",linestyle=":")
    plt.axhline(1,color="black",linestyle=":")
    plt.axvline(0.06,color="red",linestyle="--",label="bubble resonance")
    plt.xlabel(r'$ka$')
    plt.ylabel(r'$F_{bs}$')
    plt.legend()
    plt.show()

    return 0

def summingRel(n,ka):
    result = 0
    for i in n:
        result += ((2.*i)+1.)*(jn(i,ka)/hank(i,ka))*((-1.)**i)

    return result

def summingRig(n,ka):
    # F_bs Summation for Rigid Sphere, to recreate plot on p.100 and verify code is correct
    result = 0
    for i in n:
        result += ((2.*i)+1.)*(bessDer(i,ka)/hankDer(i,ka))*((-1)**i)

    return result

def bessDer(n,z):
    #return jn(n,z,True)
    result = -1*jn(n+1,z)+(n/z)*jn(n,z)
    return result

def hank(n,z):
    # Calculates the Hankel function
    result = jn(n,z)-1j*yn(n,z)
    return result

def hankDer(n,z):
    #Caclulates the first derivative of the hankel function with respect to z
    result = -1*hank(n+1,z)+(n/z)*hank(n,z)
    return result

if __name__ == "__main__":
    sys.exit(main(sys.argv))