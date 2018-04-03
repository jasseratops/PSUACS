# JAscripts
# Wave Construction
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/24/2017


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp

L = np.linspace(0,1)

def main(args):


    comp1 = compCreator(1,1)
    comp2 = compCreator(0.1,2)
    comp3 = compCreator(0.05,3)
    comp4 = compCreator(-0.1,4)

    sum = comp1 + comp2 + comp3 + comp4

    plt.figure()
    plt.plot(L,comp1)
    plt.plot(L,comp2)
    plt.plot(L,comp3)
    plt.plot(L,comp4)
    plt.plot(L,sum)
    plt.show()

    return 0

def compCreator(An,n):
    return An*sin(L*(pi)*(n))


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))