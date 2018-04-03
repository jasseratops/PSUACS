import numpy as np
import matplotlib.pyplot as plt
import math

m = 0.1
C = 25.3e-6
R = 6.28

f1 = 10
f2 = 1000

freqs = np.logspace(math.log10(f1), math.log10(f2), 201)
w = 2*math.pi*freqs

Z = (1j*w*m)+(1/(1j*w*C))+R

Y = 1/Z

# Ymag = abs(Y)
Ymag = R*abs(Y)
Yang = np.angle(Y)*(180/math.pi)

# log x axis
plt.subplot(211)
plt.loglog(freqs, Ymag)
plt.title('Magnitude of Normalized Admittance')
plt.grid(True)
plt.xlabel("Frequency (in Hz)")
plt.ylabel("Normalized Admittance")

# log x axis
plt.subplot(212)
plt.semilogx(freqs, Yang)
plt.title('Admittance Phase')
plt.grid(True)
plt.xlabel("Frequency (in Hz)")
plt.ylabel("Phase (in Degrees)")

plt.show()