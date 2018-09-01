# PSUACS
# ACS597_SigA_NoiseGen
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/30/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import random as rn


def main(args):
    V0= 4.0
    Vplus = (rn.random()*4)+1j*rn.random()*4
    Vmid = 1

    lsp = [V0,Vplus,Vmid,np.conj(Vplus)]
    N = len(lsp)

    delT = 0.01
    delF = 1/(N*delT)

    times = np.arange(0,N)*delT
    freqs = np.arange(0,N)*delF

    x_time = np.fft.ifft(lsp)

    print x_time



    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))