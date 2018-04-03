#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Underdamped.py
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


from math import sqrt, atan, exp, cos, degrees
import numpy as np
import matplotlib.pyplot as plt


def main(args):
  end = 5
  points=end*101
  
  t = np.linspace(0,end,points)
  sol = np.zeros(np.shape(t))
  decay = np.zeros(np.shape(t))
  
  x0 = 20.55E-2
  u0 = 0
  beta = 0.56
  omegaD = 8.19
  
  C = sqrt((x0**2)+(((u0+(beta*x0))/omegaD)**2))
  
  phi = atan((-u0+(beta*x0))/(x0*omegaD))
  
  
  print "C = " + str(C)
  print "phi = " + str(degrees(phi))
  
  for i in range(len(t)):
    sol[i] = C*exp(-beta*t[i])*cos((omegaD*t[i])+phi)
    decay[i] = C*exp(-beta*t[i])
    
  
    
    
  plt.figure()
  plt.plot(t,sol)
  plt.plot(t,decay)
  plt.show()
  
  return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
