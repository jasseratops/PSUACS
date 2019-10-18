# PSUACS
# localMax
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/18/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):

    a = np.array([1,2,1,4,1,6,3,2,3],dtype=float)
    print a
    print a[find_localMax(a,1)]

    return 0

def find_localMax(arr,n):
    m = 0
    arg = 0
    for max in range(n+1):
        arg = find_first_max(arr[m:])
        if m >=len(arr):
            print "No" +str(n)+ "maximum found"
            break
    return m

def find_first_max(arr):

    for i in range(len(arr)):
        if i == 0:
            print "hey"
            max = (arr[i] > arr[i+1])
        elif i == (len(arr)-1):
            max = (arr[i] > arr[i - 1])
        else:
            max = (arr[i] > arr[i-1]) and (arr[i] > arr[i+1])

        if max == True:
            return i
            break


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))