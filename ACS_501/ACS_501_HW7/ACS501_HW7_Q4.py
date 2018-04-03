# JAscripts
# ACS501_HW7_Q4
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/30/2017



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import pi, sin, cos, tan, exp, sqrt


def main(args):
    runner(1,3,1)
    runner(2,1,2)
    runner(2,3,3)

def runner(n,m,i):

    A = 1
    L = 1

    x = np.linspace(0,L,101)
    z = np.linspace(0,L,101)

    xMesh, zMesh = np.meshgrid(x, z)

    y = A*(sin((n*pi*xMesh)/L))*(sin((m*pi*zMesh)/(2*L)))
    yX = A*(sin((n*pi*x)/L))
    yZ = A*sin((m*pi*z)/(2*L))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("n=" + str(n) + ",m=" + str(m))
    ax.plot_surface(xMesh,zMesh,y, cmap=cm.coolwarm)
    ax.set_xlabel(("x [(L) m]"))
    ax.set_ylabel(("z [(L) m]"))
    ax.set_zlabel(("y [(A) m]"))
    path = 'C:/Users/alshehrj/Desktop/ACS501/HW7/Q4/' + str(i) + '-3d-' + str(n) + "x" + str(m) + ".png"
    fig.savefig(path)  # save the figure to file
    plt.close(fig)

    plt.figure()
    plt.title("2D, n=" + str(n) + ",m=" + str(m))
    plt.plot(x,yX,label="x mode")
    plt.plot(z,yZ,label="z mode")
    plt.xlabel("x,z [(L) m]")
    plt.ylabel("y [(A) m]")
    plt.legend()

    path = 'C:/Users/alshehrj/Desktop/ACS501/HW7/Q4/' + str(i) + '-2d-' + str(n) + "x" + str(m) + ".png"
    plt.savefig(path)
    plt.close(fig)

    #plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))