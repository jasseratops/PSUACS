import numpy as np
from numpy import sin, cos, tan, pi, exp

import matplotlib.pyplot as plt

c = 343.
diam1 = 12.7E-3
f = 80.
a1 = diam1/2.
k = 2*pi*f/c

x = -(2./((1./16.)-16.))*np.sqrt((10**0.6)-1.)

l = np.arcsin(x)/k
print "l: " + str(l)

print "k*l: " + str(k*l)

freq = np. linspace(20.,5000.,1024)
k2 = 2*pi*freq/c

Tw = 4/(4+(((1/16.)-16.)*sin(k2*l))**2)

f_pass = c/(2*l)
print "f_pass: " + str(f_pass)

plt.figure()
plt.title("Power Transmission Coefficient")
plt.plot(freq,Tw,label="T_w")
plt.axvline(f_pass, color="red", label="f_pass")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Transmission Power")
plt.legend()
plt.show()