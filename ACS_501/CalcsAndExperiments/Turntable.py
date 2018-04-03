#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Turntable.py
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

from math import sqrt, pi

def main(args):
  
  r = 0.25
  M = 18.11E-3
  m = 8E-3
  x = 0.04
  s = 110
  I = M*(x**2)
  
  angFreq = sqrt(-((r**2)*(m+M-s))/I)
  freq = angFreq/(2*pi)
  print I
  print angFreq
  print freq
  
  return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
