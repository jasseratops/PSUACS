# PSUACS
# ACS597_SigA_BilinearXform
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/16/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import sys


def main(args):

    return 0

def bilinearXform(nums,dens,fs):
    nn = len(nums)
    mm = len(dens)

    if nn > mm:
        sys.exit("Order of numerator cannot exceed order of denominator")
    elif nn < mm:
        padz = np.zeros(mm-nn)
        nn = np.append(nums,padz)

    rootv = -np.ones(mm-1)
    numz = np.zeros(mm)
    denz = numz

    fsfact = 1.

    for i in range(mm):
        basepoly = np.poly(rootv)*fsfact
        numz = numz + basepoly*nums[mm-i+1]
        denz = denz + basepoly*dens[mm-i+1]

        if i<(mm-1):
            fsfact=fsfact*2*fs
            rootv[i] += 2

    numz /= numz[0]
    denz /= denz[0]

    return numz, denz

if __name__ == '__main__':
    sys.exit(main(sys.argv))