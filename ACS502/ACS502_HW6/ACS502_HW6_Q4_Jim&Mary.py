import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

f = 330.
c0 = 343.
rho0 = 1.21
a = 0.01

omega = 2 * pi * f
k = omega / c0

t = np.linspace(0, 0.02, 1024)

def main(args):
    f = 330.
    c0 = 343.
    rho0 = 1.21
    a = 0.01

    omega = 2*pi*f
    k = omega/c0

    t = np.linspace(0,0.02,1024)

    print k*a

    source = np.array([90.,-17.,10.])
    jim = np.array([0.,-83.,1.7])
    mary = np.array([117.,0.,1.7])


    coords = np.stack((source,jim,mary))

    print coords
    Rj_vec = jim - source
    Rj_mag = np.linalg.norm(Rj_vec)

    Rm_vec = mary - source
    Rm_mag = np.linalg.norm(Rm_vec)

    p_rms5 = (20.E-6)*(10**(87./20.))
    A = (5.)*(np.sqrt(2))*p_rms5

    Ii_jim = I_instant(Rj_mag,A)
    Ii_mary = I_instant(Rm_mag,A)
    xs = coords[:,0]
    ys = coords[:,1]
    zs = coords[:,2]
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)
    ax.set_xlabel('East')
    ax.set_ylabel('South')
    ax.set_zlabel('Height')
    '''

    plt.figure()
    plt.subplot(211)
    plt.plot(t,np.abs(Ii_jim),label = "Jim")
    plt.plot(t,np.abs(Ii_mary),label = "Mary")
    plt.legend()

    plt.subplot(212)
    plt.plot(t,np.degrees(np.angle(Ii_jim)), label= "Jim")
    plt.plot(t,np.degrees(np.angle(Ii_mary)), label = "Mary")
    plt.legend()
    plt.show()
    return 0

def I_instant(r,A):
    Ii = (1-(1j/(k*r)))*(1/(rho0*c0))*(exp(2j*(omega*t-(k*r))))*((A/r)**2)
    return Ii


if __name__ == "__main__":
    sys.exit(main(sys.argv))