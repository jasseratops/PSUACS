# PSUACS
# Sympy_Sandbox
# Jasser Alshehri
# Starkey Hearing Technologies
# 8/28/2018


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import sympy as sp
from sympy import I
from sympy.functions import re, sqrt



def main(args):
    a, b = sp.symbols('a b')
    expr = (2*I - a*b +8)**2
    expanded = sp.expand(expr)

    print expr
    print expanded

    print re(expr)
    print re(expanded)



    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))