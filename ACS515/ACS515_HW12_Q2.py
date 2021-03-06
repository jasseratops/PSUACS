import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
from scipy.special import jv as Jn, yv as Yn, jvp as JnDer

def main(args):
    deg = 1.
    degRes = deg / 360.
    theta = pi * np.arange(0, 2, degRes)

    #ka = np.geomspace(0.01,50.,1024*2)
    ka = [0.2,2.,20.]
    N = 100

    plt.figure()
    for i in ka:
        holder = np.abs(sumPscatRelCyl(N,i,theta))
        holder /= np.max(holder)
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, holder, label="ka= " + str(i))
        ax.grid(True)
        plt.legend()
    plt.show()

    plt.show()
    return 0

def sumPscatRigCyl(N,ka,theta):
    # rigid cylinder p_scat sum, for code verification
    result = 0
    ep_m=np.ones(N)*2
    ep_m[0]=1
    for i in range(N):
        result += ep_m[i]*(JnDer(i,ka)/HankDer(i,ka))*cos(i*theta)
    return result

def sumPscatRelCyl(N,ka,theta):
    # pressure release cylinder p_scat sum
    result = 0
    ep_m=np.ones(N)*2
    ep_m[0]=1
    for i in range(N):
        result += ep_m[i]*(Jn(i,ka)/Hank(i,ka))*cos(i*theta)
    return result

def sumRigCyl(N,ka):
    # rigid cylinder back scatter sum, for code verification
    result = 0
    ep_m=np.ones(N)*2
    ep_m[0]=1
    for i in range(N):
        result += ep_m[i]*((-1)**i)*JnDer(i,ka)/HankDer(i,ka)
    return result

def Hank(n,z):
    # normal Hankel function
    result = Jn(n,z)-1j*Yn(n,z)
    return result

def HankDer(n,z):
    #Caclulates the first derivative of the hankel function with respect to z
    result = -1*Hank(n+1,z)+(n/z)*Hank(n,z)
    return result

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))