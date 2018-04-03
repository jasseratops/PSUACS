#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ComplexSolutions.py
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

from math import exp
import cmath
import numpy as np
from numpy import real
import matplotlib.pyplot as plt

def main(args):
  t = np.linspace(0,100,10100,dtype=complex)
  x = np.zeros(len(t),dtype=complex)
  y = np.zeros(len(t),dtype=complex)
  omega = 1000+1j*0
  
  for i in range(len(t)):
    print i
    x[i] = abs(20*np.exp(1j*omega*t[i]))
    y[i] = abs(20*np.exp(-1j*omega*t[i]))
    
  print x
  
  plt.figure()
  plt.plot(t,x,label="x")
  plt.plot(t,y,label="y")
  plt.legend()
  plt.show()
  
  return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
