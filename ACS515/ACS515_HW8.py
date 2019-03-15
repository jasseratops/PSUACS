# PSUACS
# ACS515_HW8
# Jasser Alshehri
# Starkey Hearing Technologies
# 3/13/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
from mpl_toolkits import mplot3d
from matplotlib import cm




def main(args):
    N = 1024
    L_x = 1.
    L_y = 1.

    x = np.linspace(-L_x/2.,L_x/2.,N)
    y = np.linspace(-L_y/2.,L_y/2.,N)

    #lim = np.max([L_x,L_y])/2.
    lim = 50.

    X, Y = np.meshgrid(x,y)

    Z = u_n(X,Y,L_x,L_y)
    print np.shape(x/L_x)
    print np.shape(Z)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    plt.xlim(lim*-1.1,lim*1.1)
    plt.ylim(lim*-1.1,lim*1.1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

    return 0

def u_n(x_s,y_s,L_x,L_y):
    return exp(-1*(((x_s/L_x)**2)+((y_s/L_y)**2)))


