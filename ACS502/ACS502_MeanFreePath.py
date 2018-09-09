import numpy as np


p0 = 1.0E5          # Pa
T = 273.0           # K
R = 8.314
Na = 6.0E23
d = 364.0E-12

lam_m = (R*T)/(np.sqrt(2)*np.pi*(d**2)*p0*Na)

print lam_m

f = 343.0/lam_m

print f