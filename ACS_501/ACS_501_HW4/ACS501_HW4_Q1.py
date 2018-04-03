# ACS501
# HW4_Q1
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/16/2017


import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, tan, exp, sqrt

stLength = np.linspace(0.5,1.5,101)
dx =np.zeros(len(stLength))
c = np.zeros(len(stLength))
T = np.zeros(len(stLength))

m = 0.235/80
L = 6E-2

# the following section was written beforethe correction from Dr. Vigeant
'''
G = 78E9
d = 1E-3
n = 80
D = 7E-2

s = (G*(d**4))/(8*n*(D**3))
print "s: ", s
'''
s = 20              #N/m

for i in range(len(stLength)):
    dx[i] = (stLength[i])/73
    c[i] = dx[i]*sqrt(s/m)
    T[i] = (stLength[i])/c[i]

print "T: ", T

plt.figure()
plt.title("Wave Speed vs. Change in Length")
plt.plot(stLength,c)
plt.xlabel(r'$L_{stretched}$ [m]')
plt.ylabel("Wave Speed [m/s]")
plt.grid(animated=True, which="both")

plt.figure()
plt.title("Propogation Time vs. Change in Length")
plt.plot(stLength,T)
plt.xlabel(r'$L_{stretched}$ [m]')
plt.ylabel("Propogation Time [s]")
plt.grid(animated=True, which="both")

plt.show()