#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Forced_Damped.py
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
from math import atan, cos, sqrt, sin, exp
import matplotlib.pyplot as plt

def main(args):
  endTime = 10
  points = 101*endTime
  
  t = np.linspace(0,endTime,points)
  xt_ss = np.zeros(np.shape(t))
  xt_ss_noTheta = np.zeros(np.shape(t))
  xt_tran = np.zeros(np.shape(t))
  xt_comp = np.zeros(np.shape(t))
  ut_ss = np.zeros(np.shape(t))
  at_ss = np.zeros(np.shape(t))
  Ft = np.zeros(np.shape(t))
  
  
  
  angFreq = np.logspace(0,2,1000)
  
  xRes = np.zeros(np.shape(angFreq))
  NORMxRes = np.zeros(np.shape(angFreq))
  #given
  s = 100
  m = 0.5
  Rm = 1.4
  
  
  F0=2
  omega0 = sqrt(s/m)
  omega = 5
  beta = Rm/(2*m)
  
  
  Cprime = 0.05
  omegaD = sqrt((omega0**2)-(beta**2))
  print "Omega0: ", omega0
  print "OmegaD: ", omegaD
  
  X_ss = (F0/m)/sqrt((((omega0**2)-(omega**2))**2) + ((2*beta*omega)**2))
  print "X_ss = ", X_ss
  theta = atan((2*beta*omega)/((omega0**2)-(omega**2)))
  print "theta = ", np.degrees(theta), "degrees"
  
  PIavg = (((omega**2)*beta*(F0**2))/m)/((((omega**2)-(omega0**2))**2)+((2*beta*omega)**2))
  print "Avg. Power = ", "%.2E"%PIavg, " Watts"
  
  
  for i in range(len(t)):
    Ft[i] = 2*cos(omega*t[i])
    xt_ss[i] = X_ss*cos((omega*t[i])+theta)
    xt_ss_noTheta[i] = X_ss*cos(omega*t[i])
    xt_tran[i] = Cprime*exp(-beta)*cos(omegaD*t[i])
    xt_comp[i] = xt_tran[i] + xt_ss_noTheta[i]
    ut_ss[i] = -omega*X_ss*sin((omega*t[i])+theta)
    at_ss[i] = -(omega**2)*X_ss*cos((omega*t[i])+theta)
    
  for i in range(len(angFreq)):
    xRes[i] = (F0/m)/sqrt((((omega0**2)-(angFreq[i]**2))**2) + ((2*beta*angFreq[i])**2))
    
  normVal = max(xRes)
  resFreq = angFreq[np.argmax(xRes)]
  print "resFreq: ", resFreq
  
  for i in range(len(xRes)):
    NORMxRes[i] = xRes[i]/normVal
    
  
    
  plt.figure()
  plt.title("Steady State Displacement, Velocity, Acceleration, and Driving Force")
  plt.plot(t,Ft,label= "Driving Force")
  plt.plot(t,xt_ss,label= "Displ. (SS)")
  plt.plot(t,ut_ss,label= "Vel. (SS)")
  plt.plot(t,at_ss,label= "Accel. (SS)")
  plt.grid(animated = True,which="both") 
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude (various units)")
  plt.legend()
  
  
  plt.figure()
  plt.title("Displacement Solutions")
  plt.plot(t,xt_tran,label= "Displ. (Tran)")
  plt.plot(t,xt_ss, label= "Displ. (SS)")
  plt.plot(t,xt_comp, label= "Displ. (Total)")
  plt.grid(animated = True,which="both") 
  plt.xlabel("Time (s)")
  plt.ylabel("Displacement Amplitude (m)")
  plt.legend()
  
  plt.figure()
  plt.title("Displacement vs. Frequency")
  plt.semilogx(angFreq,NORMxRes,label= "Displ. Ampl.")
  plt.axvline(x=omega0, ls='--',label= "Natural Freq., Resonance", color= "r")
  plt.axvline(x=omega, label= "Driving Freq.", color= "orange")
  plt.axvline(x=omegaD, label= "Natural DAMPED Freq.", color= "g")
  plt.xlim(angFreq[0],angFreq[-1])
  plt.grid(animated = True,which="both") 
  plt.xlabel("Angular Frequency (rad/s)")
  plt.ylabel("Displacement Amplitude (m)")
  plt.legend()
  
  
  plt.show()
  
  
  
  return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
