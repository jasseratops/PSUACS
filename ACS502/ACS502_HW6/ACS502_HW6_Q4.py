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
h=10

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
    jim = np.array([0.,-83.,20])
    mary = np.array([117.,0.,20])

    o_prime = np.array([90.,-17.,0])


    coords = np.stack((source,jim,mary))

    print coords
    Rj_vec = jim - source
    Rj_mag = np.linalg.norm(Rj_vec)

    Rm_vec = mary - source
    Rm_mag = np.linalg.norm(Rm_vec)

    p_rms5 = (20.E-6)*(10**(87./20.))
    print p_rms5
    A = (5.)*(np.sqrt(2))*p_rms5
    print A
    Ii_jim = I_instant(Rj_mag,A)
    Ii_mary = I_instant(Rm_mag,A)
    xs = coords[:,0]
    ys = coords[:,1]
    zs = coords[:,2]

    Rj_vec_R = jim - o_prime
    Rj_mag_R = np.linalg.norm(Rj_vec)
    theta_j = vecAngle(Rj_vec_R)

    Rm_vec_R = mary - o_prime
    Rm_mag_R = np.linalg.norm(Rm_vec)
    theta_m = vecAngle(Rm_vec_R)

    Ii2_jim = I_inst2(Rj_mag_R,A,theta_j)
    Ii2_mary = I_inst2(Rm_mag_R,A,theta_m)

    print theta_j
    print theta_m

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('East')
    ax.set_ylabel('South')
    ax.set_zlabel('Height')


    plt.figure()
    plt.plot(t,Ii_jim,label = "Jim")
    plt.plot(t,Ii_mary,label = "Mary")
    plt.xlim(t[0],t[-1])
    plt.title("without Reflection")
    plt.xlabel("time [s]")
    plt.ylabel(r"$I_i$ [W/${m^2}$]")
    plt.legend()

    plt.figure()
    plt.plot(t,Ii2_jim,label = "Jim")
    plt.plot(t,Ii2_mary,label = "Mary")
    plt.xlim(t[0],t[-1])
    plt.title("with Reflection")
    plt.xlabel("time [s]")
    plt.ylabel(r"$I_i$ [W/${m^2}$]")
    plt.legend()
    plt.show()
    return 0

def I_instant(r,A):
    Ii = (1/(rho0*c0))*(((A*cos(omega*t-(k*r)))/r)**2)
    return Ii

def I_inst2(r,A,theta):
    Ii2 = (((A**2)/(2*r**2))/(rho0*c0))*(4*(cos(k*h*cos(theta))**2))*exp(2j*(omega*t-k*r))
    return Ii2

def vecAngle(vec):
    unit = np.array([0,0,1])
    theta = np.arccos(np.dot(unit,vec)/np.linalg.norm(vec))
    return theta

if __name__ == "__main__":
    sys.exit(main(sys.argv))