#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Damping.py
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

import math
import numpy as np
import matplotlib.pyplot as plt

def main(args):
  t = np.linspace(0,5,10100)
  w0 = 100.00
  beta = 150.00
  wd = np.sqrt(abs(w0**2 - beta**2))
  A = 1
  phi = 0
  x = np.zeros(np.shape(t))
  
  print (beta/w0)
  
  for i in range(len(t)):
    x[i] = A*(np.exp(-beta*t[i]))*math.cos(wd*t[i]+phi)
  
  
  plt.figure()
  plt.plot(t,x)
  plt.show()
  
  return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
