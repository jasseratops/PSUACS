# PSUACS
# ACS502_HW2_Q3
# Jasser Alshehri
# Starkey Hearing Technologies
# 9/10/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp, genfromtxt

foldername = "//Starfile/Dept/Engineering/Applications/Application Eng Data/JasserA/PSU/ACS502 - Elements of Waves in Fluids/"
filename = "ACS502_HW2_Q3_Table.csv"
def main(args):
    # bring in values from table
    path = foldername + filename
    dat = genfromtxt(path, delimiter=',')

    m = 184.41E-3
    diam = 44.33E-3
    r = diam/2.0
    A = pi*(r**2)
    P0 = 97.3E3
    g = 9.8

    h_i = dat[1:, 0]
    T_i = dat[1:, 1]

    print h_i
    print T_i

    # convert T_i from ms to s

    T_i *= 1.0E-3

    print T_i

    # square T_i

    T_i2 = T_i**2
    print T_i2

    C = ((P0*A/m)+g)/(4.0*(pi**2))

    print C

    CT_i2 = C*T_i2

    print CT_i2

    #slope = (max(T_i2)-min(T_i2))/(max(h_i)-min(h_i))

    plt.figure()
    plt.scatter(h_i,CT_i2)
    plt.xlabel("h_i")
    plt.ylabel("T_i^2")
    plt.show()


    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))