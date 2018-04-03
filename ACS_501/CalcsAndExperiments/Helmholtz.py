#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Helmholtz.py
#  
#  Copyright 2017 Jasser Alshehri <jasser_alshehri@starkey.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import numpy as np
import matplotlib.pyplot as plt
from math import pi,cos,sin,sqrt
from winsound import Beep

c = 343

# Given
L1 = 4.5E-2
a1 = 0.8E-2
w1 = 5.8E-2
h1 = 11.2E-2

# Calculated
S1 = (pi*(a1**2))
V1 = (w1**2)*h1
Leff1 = L1+(1.4*a1)

print "S1: " + str('%.2E'%S1)
print "V1: " + str('%.2E'%V1)
print "Leff1: " + str('%.2E'%Leff1)

# Given
L2 = 9.5E-2
a2 = 1.1E-2
r2 = 4.4E-2

# Calc
S2 = (pi*(a2**2))
V2 = (4/3)*pi*(r2**3)
Leff2 = L2+(1.4*a2)

print "S2: " + str('%.2E'%S2)
print "V2: " + str('%.2E'%V2)
print "Leff2: " + str('%.2E'%Leff2)


def findResFreq(S,Leff,V):
  omega0 = c*sqrt(S/(Leff*V))
  return omega0

def main(args):
  omega01 = findResFreq(S1,Leff1,V1)
  omega02 = findResFreq(S2,Leff2,V2)
  
  print "Resonant Frequency of Bottle #1: " + str(omega01)+"Hz"
  Beep(int(omega01),1000)
  print "Resonant Frequency of Bottle #2: " + str(omega02)+"Hz"
  Beep(int(omega02),1000)
  
  print "Difference in frequency: " + str(abs(omega01-omega02))
  
  return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
