# PSUACS
# ACS515_HW8
# Jasser Alshehri
# Starkey Hearing Technologies
# 3/13/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
from mpl_toolkits import mplot3d



def main(args):
    N = 1024
    L_x = 400.
    L_y = 400.

    #x = np.linspace(-L_x/2.,L_x/2.,N)
    #y = np.linspace(-L_y/2.,L_y/2.,N)

    x = np.linspace(0,3*L_x,N)
    y = np.linspace(0,3*L_y, N)

    X, Y = np.meshgrid(x,y)

    Z = u_n(X/L_x,Y/L_y)
    print np.shape(x/L_x)
    print np.shape(Z)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');

    plt.show()

    return 0

def u_n(x_s,y_s):
    return exp(-1*(x_s+y_s))



if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))