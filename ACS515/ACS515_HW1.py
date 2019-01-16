# PSUACS
# ACS515_HW1
# Jasser Alshehri
# Starkey Hearing Technologies
# 1/15/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):

    x = np.linspace(-3.,3.,1024)
    t = 0
    rho0 = 1.21
    c0 = 343.
    P0 = 1.01325E5
    T0 = 293.
    gam = 1.4

    Phi = exp(-((x-(343.*t))**2))
    u = -2*(x-(c0*t))*Phi
    p = -rho0*2*c0*(x-(c0*t))*Phi
    rho = -(rho0/c0)*2*(x-(c0*t))*Phi
    T = -rho0*((gam-1)/gam)*(T0/P0)*2*c0*(x-(c0*t))*Phi


    Phi_N = exp(-((x+(343.*t))**2))
    u_N = -2*(x+(c0*t))*Phi_N
    p_N = rho0*2*c0*(x+(c0*t))*Phi_N
    rho_N = (rho0/c0)*2*(x+(c0*t))*Phi_N
    T_N = rho0*((gam-1)/gam)*(T0/P0)*2*c0*(x+(c0*t))*Phi

    plt.figure()
    plt.plot(x,Phi)
    plt.xlim(x[0], x[-1])
    plt.title("Velocity Potential")
    plt.xlabel("Position [m]")
    plt.ylabel("Velocity Potential")
    plt.grid()

    plt.figure()
    plt.subplot(411)
    plt.plot(x, u)
    plt.xlim(x[0],x[-1])
    plt.ylabel("m/s")
    plt.title("Particle Velocity")
    plt.grid()
    plt.subplot(412)
    plt.plot(x, p)
    plt.xlim(x[0], x[-1])
    plt.ylabel("Pa")
    plt.title("Pressure")
    plt.grid()
    plt.subplot(413)
    plt.plot(x, rho)
    plt.xlim(x[0], x[-1])
    plt.ylabel("kg/m^3")
    plt.title("Acoustic Density")
    plt.grid()
    plt.subplot(414)
    plt.plot(x, T)
    plt.xlim(x[0], x[-1])
    plt.ylabel("K")
    plt.title("Temperature")
    plt.subplots_adjust(hspace=0.8,left=0.17)
    plt.xlabel("Position [m]")
    plt.grid()

    plt.figure()
    plt.plot(x,Phi_N)
    plt.xlim(x[0], x[-1])
    plt.title("Velocity Potential, Negative x direction")
    plt.xlabel("Position [m]")
    plt.ylabel("Velocity Potential")
    plt.grid()

    plt.figure()
    plt.subplot(411)
    plt.plot(x, u_N)
    plt.xlim(x[0], x[-1])
    plt.ylabel("m/s")
    plt.title("Particle Velocity")
    plt.grid()
    plt.subplot(412)
    plt.plot(x, p_N)
    plt.xlim(x[0], x[-1])
    plt.ylabel("Pa")
    plt.title("Pressure")
    plt.grid()
    plt.subplot(413)
    plt.plot(x, rho_N)
    plt.xlim(x[0], x[-1])
    plt.ylabel("kg/m^3")
    plt.title("Acoustic Density")
    plt.grid()
    plt.subplot(414)
    plt.plot(x, T_N)
    plt.xlim(x[0], x[-1])
    plt.ylabel("K")
    plt.title("Temperature")
    plt.grid()
    plt.subplots_adjust(hspace=0.8,left=0.17)
    plt.xlabel("Position [m]")
    plt.show()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))