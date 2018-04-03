import numpy as np
import matplotlib.pyplot as plt
import math

R = 0.01
C = 0.1
f1 = 10
f2 = 1000
freqs = np.logspace(math.log10(f1),math.log10(f2),201)
w = 2*math.pi*freqs

zArray = R + 1/(1j*w*C)       # Impedance for B5
# zArray = (R/(1+(R*C*w*1j))) # Impedance for B6

zMag = abs(zArray)
zAng = np.angle(zArray)*(180/math.pi)

# log x axis
plt.subplot(211)
plt.loglog(freqs, zMag)
plt.title('Z Mag')
plt.grid(True)
plt.xlabel("Frequency (in Hz)")
plt.ylabel("Impedance (in Ns/m)")

# log x axis
plt.subplot(212)
plt.semilogx(freqs, zAng)
plt.title('Z Phase')
plt.grid(True)
plt.xlabel("Frequency (in Hz)")
plt.ylabel("Phase (in Degrees)")

plt.show()
