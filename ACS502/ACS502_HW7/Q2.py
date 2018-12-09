# PSUACS
# PolarTest
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/20/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):

    deg = 0.01
    degRes = deg/360.
    theta = np.pi * np.arange(0,2,degRes)
    '''
    a = 1.
    dir = (abs(cos(theta)))
    f = 10.
    c = 343.
    omega = 2*pi*f
    k = omega/c
    d = 4
    amp = 1.
    print k*d

    kd = pi*1.8
    psi = pi/4.
    p = abs(cos((kd/2)*sin(theta) + psi))
    SPL = 20*np.log10(abs(p))
    '''
    N = 3.
    c = 1500.
    f = 10.E3
    k = 2*pi*f/c
    d = 0.075
    psi = pi

    #psi = 0
    f_harm = 20.E3
    k_harm = 2*pi*f_harm/c

    Dir2 = (2.-cos(k*d*cos(theta)))/2.
    Dir2_log = 10*np.log10(np.abs(Dir2))
    print Dir2


    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, Dir(k,d,theta),label="10kHz")
    ax.plot(theta, Dir(k_harm, d, theta),label="20kHz")
    #ax.set_rmax(max(SPL)+1)
    #ax.set_rticks([0.25, 0.5,0.75, 1])  # less radial ticks
    #ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)

    ax.set_title("Q2: 10log10(D(theta))", va='bottom')
    plt.legend()
    plt.show()
    return 0

def Dir(k,d,theta):
    Dir = (2. - cos(k * d * cos(theta))) / 2.
    Dir_log = 20*np.log10(np.abs(Dir))

    return Dir
if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))