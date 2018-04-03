import numpy as np
import math
import matplotlib.pyplot as plt

freqs = np.linspace(1,100,100)
R = 1
C1 = 0.01
C2 = 0.01

def twoImp(freqs):
    w = freqs*math.pi*2

    Za = R
    Zb = 1/(1j*w*C1)
    Zl = 1/(1j*w*C2)

    T = np.matrix([[1+(Za/Zb), -Za],[(1/Zb), -1]])
    F2 = np.matrix([1,0])*T*np.matrix([[-Zl],[1]])
    v2 = np.matrix([0,1])*T*np.matrix([[-Zl],[1]])
    Z2 = F2/v2
    return Z2

Imp = twoImp(freqs)
print(str(Imp))
print(len(Imp))
print(len(freqs))

# log x axis
plt.subplot(211)
plt.loglog(freqs, abs(Imp))
plt.title('Z Mag')
plt.grid(True)
plt.xlabel("Frequency (in Hz)")
plt.ylabel("Impedance (in Ns/m)")

# log x axis
plt.subplot(212)
plt.semilogx(freqs, np.angle(Imp)*(180/math.pi))
plt.title('Z Phase')
plt.grid(True)
plt.xlabel("Frequency (in Hz)")
plt.ylabel("Phase (in Degrees)")

plt.show()