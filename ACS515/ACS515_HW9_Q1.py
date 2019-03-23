# PSUACS
# ACS515_HW6_Q2
# Jasser Alshehri
# Starkey Hearing Technologies
# 2/19/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    deg = 0.01
    degRes = deg/360.
    theta = np.pi * np.arange(0,2,degRes)
    R_array = np.array([0.1,2.5,10])

    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, np.abs(Hyper1(theta)))
    ax.grid(True)
    plt.savefig("ACS515_HW9_Q1_A")

    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, np.abs(Hyper2(theta)))
    ax.grid(True)
    plt.savefig("ACS515_HW9_Q1_B")

    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, np.abs(Hyper3(theta)))
    ax.grid(True)
    plt.savefig("ACS515_HW9_Q1_C")
    plt.show()
    return 0

def Hyper1(theta):
    a = [1./3.,2./3.,0,0,0]
    Dir = a[0]*P0(theta)+a[1]*P1(theta)+a[2]*P2(theta)+a[3]*P3(theta)+a[4]*P4(theta)
    print np.degrees(theta[np.argmin(np.abs(Dir))])
    return Dir

def Hyper2(theta):
    a = [2./30.,2./5.,8./15.,0,0]
    Dir = a[0]*P0(theta)+a[1]*P1(theta)+a[2]*P2(theta)+a[3]*P3(theta)+a[4]*P4(theta)
    print np.degrees(theta[np.argmin(np.abs(Dir))])
    return Dir

def Hyper3(theta):
    a = [4./21.,4./35.,0.380952,16./35.,0]
    Dir = a[0]*P0(theta)+a[1]*P1(theta)+a[2]*P2(theta)+a[3]*P3(theta)+a[4]*P4(theta)
    print np.degrees(theta[np.argmin(np.abs(Dir))])
    return Dir


def P0(theta):
    return np.ones_like(theta)
def P1(theta):
    return cos(theta)
def P2(theta):
    return 0.5*((3*cos(theta)**2)-1)
def P3(theta):
    return 0.5*(5*(cos(theta)**3)-(3*cos(theta)))
def P4(theta):
    return (1./8.)*(35.*cos(theta)**4 - 30.*cos(theta)**2 + 3.)

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))