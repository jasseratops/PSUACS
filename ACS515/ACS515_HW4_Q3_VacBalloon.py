# PSUACS
# ACS515_HW4_Q3_VacBalloon
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/7/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    a = 0.5
    mu = 8435.
    V = (4./3.)*pi*(a**3)
    rho0 = 1.225
    m_ball = 0.5
    g = 9.81
    S = 4*pi*(a**2)
    c = 343.
    kg_of_air = V*rho0
    print kg_of_air

    h = -mu*np.log(m_ball/(kg_of_air))
    print h
    rhoAir = rho0*np.exp(-h/mu)
    s_eff = (g/mu)*V*rhoAir
    Mrad = V*rhoAir*3.
    M_eq = Mrad+m_ball

    omega = np.sqrt(s_eff/M_eq)

    print omega
    f = omega/(2*pi)
    print f
    k = omega/c
    ka = k*a
    print "ka: " + str(ka)


    R_rad = rhoAir*c*S*((ka**2)/(1+(ka**2)))
    beta = R_rad/(2*M_eq)

    tau = 1./beta
    print tau
    eta = 1.72E-5
    R_visc = 6*pi*eta*a

    gamma = R_visc/(2*M_eq)

    beta_eq = beta+gamma
    tau_eq = 1./beta_eq

    print tau_eq

    # http://hyperphysics.phy-astr.gsu.edu/hbase/oscda.html
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))