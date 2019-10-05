# PSUACS
# ACS519_Proj1_Q1
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/5/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import ACS519_StructInt.soundStructInt as ssint

def main(args):
    E = 68.9E9      # Pa
    v = 0.33        #
    rho = 2.7E3     # kg/m^3
    eta = 0.004
    h = 6.35E-3     # m
    a = 22.5E-2     # m
    b = 12.4E-2     # m

    m_length = 10
    n_length = 10

    table = np.zeros((m_length*n_length,3),dtype=complex)

    D_comp = ssint.flex_rig(E,h,v,eta)

    for m in range(m_length):
        for n in range(n_length):
            omega_comp = ssint.ss_flatplate_frq(D_comp,rho,h,m,a,n,b)
            ind = (m*10) + n
            table[ind]= omega_comp,m,n

    low10 = np.sort(table[:,0])[:11]

    print "|m|n|omega |"
    for x in range(len(low10)):
        compare = low10[x]
        for i in range(len(table[:,0])):
            if table[i,0] == compare:
                omega = table[i,0]
                m = str(int(table[i,1].real))
                n = str(int(table[i,2].real))
                print "|" + m + "|" + n + "|" + str(omega)















    return 0





if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))