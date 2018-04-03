import numpy as np
import scipy.special as spec
from numpy import pi, cos, sin, exp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


A = 1
theta = np.linspace(0,2*pi,100)



def main(args):

    A = 1

    generator(1, 1, 1.84)
    generator(2, 1, 3.05)
    generator(0, 2, 3.83)


def generator(m,n,jmn):

    kr = np.linspace(0,jmn,100)
    x = jmn*cos(theta)
    y = jmn*sin(theta)
    xMesh, yMesh = np.meshgrid(x,y)

    '''
    y = A*spec.jv(m,kr)
    yRot = A*spec.jv(m,PHI)*cos(m*THETA)
    '''

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)


    plt.figure()
    plt.plot(kr,y)
    plt.show()




if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
