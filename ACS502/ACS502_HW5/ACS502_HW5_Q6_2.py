# PSUACS
# ACS502_HW5_Q6
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/30/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    mu = 1.56
    Cp = 0.14
    kappa = 8.21
    mu_B = 2.03
    gamma = 1.13
    c0 = 1450.
    rho0 = 13600.
    V_tilde = ((4/3)*mu) + mu_B

    tau_th = ((gamma - 1)*kappa)/(rho0*(c0**2)*Cp)
    tau_v = V_tilde/(rho0*(c0**2))

    tau_tv = tau_th+tau_v

    alpha = (2*(pi**2)/c0)*tau_tv

    print "alpha: " + str(alpha) + " Nepers/(m.Hz^2)"

    omegaM = 1./tau_tv

    fM = omegaM/(2*pi)
    print omegaM
    print "Freq: " + str(fM) + " Hz"

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
