import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt




class testFuncs:
    def __init__(self):
        return None

    def func1(self):
        print str(1+1)
        return 0

    def func2(self):
        print "hi"
        return 8

def main(args):
    a = testFuncs()
    a.func1()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))