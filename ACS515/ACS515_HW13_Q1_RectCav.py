import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt


def main(args):
    c = 343.
    L_x = 9.75
    L_y = 7.75
    L_z = 2.75

    M = 5
    sh = (M,M,M)
    A = np.zeros(sh)

    for n_x in range(M):
        for n_y in range(M):
            for n_z in range(M):
                f = (c/2.)*np.sqrt(((n_x/L_x)**2)+((n_y/L_y)**2)+((n_z/L_z)**2))
                A[n_x,n_y,n_z]= f

    B = np.sort(A,axis=None)


    counter = 1
    for n_x in range(M):
        for n_y in range(M):
            for n_z in range(M):
                #print "(" + str(n_x) + "," + str(n_y) + "," + str(n_z) + "): " + str(A[n_x,n_y,n_z])
                if A[n_x,n_y,n_z] in B[1:26]:
                    print "-"*20
                    print str(counter) + ": " + str(A[n_x,n_y,n_z])
                    print "(" + str(n_x) + "," + str(n_y) + "," + str(n_z) + ")"
                    nonzer = np.count_nonzero([n_x,n_y,n_z])
                    if nonzer == 1:
                        print "Axial"
                    elif nonzer == 2:
                        print "Tangential"
                    else:
                        print "Oblique"
                    counter +=1

    N_band_250 = modesInBandRect(176.,353.,c,L_x,L_y,L_z)
    N_band_1000= modesInBandRect(707.,1414.,c,L_x,L_y,L_z)
    print "-"*100
    print "Number of modes in 250_octave band: " + str(N_band_250)
    print "Number of modes in 1000_octave band: " + str(N_band_1000)

    return 0

def modesInBandRect(f1,f2,c,L_x,L_y,L_z):
    V = L_x*L_y*L_z
    S = (L_x*L_y + L_y*L_z + L_z*L_x)*2
    L = (L_x+L_y+L_z)*4
    N_top = ((4*pi*V/(c**3))*((f2**3)/3.))+((pi*S/(2*(c**2)))*((f2**2)/2.))+((L/(8*c))*f2)
    N_bot = ((4*pi*V/(c**3))*((f1**3)/3.))+((pi*S/(2*(c**2)))*((f1**2)/2.))+((L/(8*c))*f1)
    N_band = N_top-N_bot
    return N_band

if __name__ == "__main__":
    sys.exit(main(sys.argv))