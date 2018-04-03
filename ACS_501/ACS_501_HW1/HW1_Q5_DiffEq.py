#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  HW1_Q4_DiffEq.py
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

from math import sqrt, exp

def main(args):
  r = (5+sqrt(73))/4
  
  A = 1
  t = 0
  
  x  = A*exp(r*t)
  xd = r*A*exp(r*t)
  xdd= (r**2)*A*exp(r*t)
  
  y = (2*xdd) - (5*xd) - (6*x) 
  z = (2*(r**2))-(5*r)-6

  print y
  print z  
  
  return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
