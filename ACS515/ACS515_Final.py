# PSUACS
# ACS515_Final
# Jasser Alshehri
# Starkey Hearing Technologies
# 4/27/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp

c = 343.
f = 1000.

def main(args):
    ### Rectangle
    L_x = 9.14
    L_y = 4.57
    L_z = 3.05

    V_rect = L_x*L_y*L_z
    S_rect = (L_x*L_x + L_y*L_y + L_x*L_z)*2.
    L_rect = (L_x+L_y+L_z)*4.

    print "Rectangle"
    print "V: " + str(V_rect)
    print "S: " + str(S_rect)
    print "L: " + str(L_rect)

    ### Cylinder
    r_cyl = 3.65
    h_cyl = L_z
    V_cyl = (pi*r_cyl**2)*(h_cyl)
    S_cyl = (2*pi*r_cyl)*h_cyl + 2.*(pi*r_cyl**2)
    L_cyl = (4*pi*r_cyl)+4*h_cyl

    print "-"*10
    print "Cylinder"
    print "V: " + str(V_cyl)
    print "S: " + str(S_cyl)
    print "L: " + str(L_cyl)


    ### Sphere
    r_sph = 3.12
    V_sph = (4./3.)*pi*r_sph**3
    S_sph = 4.*pi*r_sph**2
    L_sph = 2*pi*r_sph

    print "-"*10
    print "Sphere"
    print "V: " + str(V_sph)
    print "S: " + str(S_sph)
    print "L: " + str(L_sph)

    ###
    print "-"*10

    N_rect = ModalDens(V_rect,S_rect,L_rect)
    N_cyl = ModalDens(V_cyl,S_cyl,L_cyl)
    N_sph = ModalDens(V_sph,S_sph,L_sph)

    print "N_rect = " + str(N_rect)
    print "N_cyl = " + str(N_cyl) +", " + str(N_cyl/N_rect) + "%"
    print "N_sph = " + str(N_sph) + ", " + str(N_sph/N_rect) + "%"

    return 0

def ModalDens(V,S,L):
    N3d = (4.*pi*V/3.)*(f/c)**3 + (pi*S/4.)*(f/c)**2 + (L*f/(8.*c))

    return N3d

if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))