import numpy as np
import matplotlib.pyplot as plt
import math

Omega = np.logspace(math.log10(0.1),math.log10(10),101)
R = 6.28

f1 = 10
f2 = 1000

def normY(Q,Omega):
    Ynormd = ((1j*Omega)/Q)/(1-Omega**2+((1j*Omega)/Q))
    return Ynormd


# log x axis
plt.subplot(211)
plt.loglog(Omega, abs(normY(10,Omega)))
plt.loglog(Omega, abs(normY(1,Omega)))
plt.title('Magnitude of Normalized Admittance')
plt.grid(True)
plt.xlabel("Omega")
plt.ylabel("Normalized Admittance")

# log x axis
plt.subplot(212)
plt.semilogx(Omega, np.angle(normY(10,Omega)))
plt.semilogx(Omega, np.angle(normY(1,Omega)))
plt.title('Admittance Phase')
plt.grid(True)
plt.xlabel("Omega")
plt.ylabel("Phase (in Degrees)")

plt.show()