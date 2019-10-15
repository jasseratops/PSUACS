# PSUACS
# soundStructInt
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/5/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def plate_phase_speed(h,v,E,rho,omega,K):
    I_prime = plate_mom_inertia_wid(h)
    G = structural_shear_mod(E,v)
    D = flex_rig(E,h,v)
    KhG = K*h*G
    D_KhG = D/KhG

    A = np.sqrt(((D_KhG-(I_prime/h))**2)*(omega**4) + 4*(D/(rho*h))*(omega**2))
    B = (omega**2)*(D_KhG + (I_prime/h))
    C = 2.*(1-(omega**2)*(I_prime*rho/KhG))

    c_B = np.sqrt((A-B)/C)
    c_B_Mindlin = ((D/(rho*h))*(omega**2))**(1./4.)
    c_B_thin = np.ones_like(omega)*shear_speed(E,rho,K,v)
    return c_B, c_B_Mindlin, c_B_thin

def shear_speed(E,rho,K,v):
    G = structural_shear_mod(E,v)
    c_S = np.sqrt(K * G / rho)
    return c_S

def structural_shear_mod(E,v):
    G = E / (2. * (1. + v))
    return G

def flex_rig(E,h,v,eta = 0.):
    I_prime = plate_mom_inertia_wid(h)
    D = E * (h ** 3) / (12. * (1. - (v ** 2)))
    D_comp = D*(1 + (-1j*eta))
    return D_comp

def plate_mom_inertia_wid(h):
    I_prime = (h**3)/12.
    return I_prime

def ss_flatplate_frq(D_comp,rho,h,m,a,n,b):
    omega_comp = np.sqrt(D_comp/(rho*h))*(((m*pi/a)**2)+(n*pi/b)**2)
    return omega_comp