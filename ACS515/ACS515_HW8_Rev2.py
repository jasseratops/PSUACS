# PSUACS
# ACS515_HW8_Rev2
# Jasser Alshehri
# Starkey Hearing Technologies
# 3/15/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
from mpl_toolkits import mplot3d
from matplotlib import cm

L_x = 30.
L_y = 30.
f = 1000.

c = 0.466
a = 2. / (c * L_x)
b = 2. / (c * L_y)
lim = 50.


def main(args):
    #velo()
    dire()
    plt.show()
    return 0

def velo():
    N = 1024
    x = np.linspace(-L_x/2.,L_x/2.,N)
    y = np.linspace(-L_y/2.,L_y/2.,N)


    X, Y = np.meshgrid(x,y)
    u_n=exp(-1*((a*X)**2))*exp(-1*((b*Y)**2))

    Z = u_n

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    plt.xlim(lim*-1.1,lim*1.1)
    plt.ylim(lim*-1.1,lim*1.1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def dire():
    omega = 2.*pi*f
    k = omega/c

    thet = np.linspace(0.,2*pi,1024)
    phi = np.linspace(0.,pi,1024)
    THET,PHI = np.meshgrid(thet,phi)

    k_x = k*sin(THET)*cos(PHI)
    k_y = k*sin(THET)*sin(PHI)
    dir = exp(-1*((c ** 2.) / 16.)*((L_x*k_x) + (L_y*k_y)))

    R = np.abs(dir)

    X = R*sin(THET)*cos(PHI)
    Y = R*sin(THET)*sin(PHI)
    Z = R


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    plt.xlim(lim * -1.1, lim * 1.1)
    plt.ylim(lim * -1.1, lim * 1.1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))


