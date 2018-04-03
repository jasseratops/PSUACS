import numpy as np
from numpy import sin, cos, pi
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(args):

    theta = np.linspace(0,2*pi,100)
    r = 2


    x = r*cos(theta)
    y = r*sin(theta)

    xMesh, yMesh = np.meshgrid(x,y)

    z = np.sqrt((xMesh**2) + (yMesh**2))
    print "here"

    fig = plt.figure()
    ax = fig.gca(projection='polar')
    print ax
    surf = ax.plot_surface(xMesh, yMesh, z, cmap=cm.coolwarm)
    plt.show()


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))