import numpy as np
from numpy import log10, pi

def main():
    rho0 = 13600.
    gamma = 1.13
    c0 = 1450.
    mu = 1.56
    Cp = 0.14
    kappa = 8.21
    muB = 2.03

    V_tilde = (4./3.)+(muB/mu)

    Pr = mu*Cp/kappa

    alpha_tv = (mu/(2*rho0*(c0**2)))*(V_tilde + ((gamma-1)/Pr))*(4*(pi**2))

    print alpha_tv
    return 0

main()