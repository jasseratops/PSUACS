import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
from mpl_toolkits import mplot3d
from matplotlib import cm

L_x = 1.
L_y = 5.
f = 2.
c_0 = 343.
omega = 2.*pi*f
k = omega/c_0
c = 0.466

N = 1024

def main(args):
    #velo(L_x_inp=0.5,L_y_inp=0.5)
    #velo(L_x_inp=4.,L_y_inp=1.)
    #velo(L_x_inp=1,L_y_inp=10.)

    dire(k_inp=2)
    #dire(k_inp=10)
    #dire(k_inp=20)

    plt.show()
    return 0

def velo(L_x_inp = L_x, L_y_inp = L_y):
    L_x = L_x_inp
    L_y = L_y_inp
    lim = np.max([L_x, L_y]) / 2.
    a = 2. / (c * L_x)
    b = 2. / (c * L_y)

    x = np.linspace(-L_x / 2., L_x / 2., N)
    y = np.linspace(-L_y / 2., L_y / 2., N)

    X, Y = np.meshgrid(x, y)

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

def dire(k_inp=k):
    k = k_inp

    thet = np.linspace(0.,pi,1024)
    phi = np.linspace(0.,2.*pi,1024*2)

    THET,PHI = np.meshgrid(thet,phi)

    k_x = k*sin(THET)*cos(PHI)
    k_y = k*sin(THET)*sin(PHI)
    dir = exp(-1*((c ** 2.) / 16.)*((L_x*k_x) + (L_y*k_y)))

    R= dir

    X = R*sin(THET)*cos(PHI)
    Y = R*sin(THET)*sin(PHI)
    Z = np.abs(R*cos(THET))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,label=str(k))
    ax.set_title("k="+str(k))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))