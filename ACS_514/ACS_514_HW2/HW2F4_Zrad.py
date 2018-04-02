import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spsp
import scipy

ka = np.linspace (0.1,10,100)                                                     # Define ka (normalized frequency)

print(len(ka))

Zim = Zre = zRadExact = zRadLFapprox = zRadParApprox = np.array([])               # Initialize Zrad




# Z Exact Function
for i in range(len(ka)):

    # Z (Low Frequency Approximation) Fill-Up
    zRadLFresult = (((ka[i])**2)/2) + (1j*(8/(3*math.pi))*ka[i])
    zRadLFapprox = np.append(zRadLFapprox,zRadLFresult)

    # Z (Parallel Approximation) Fill-Up
    zRadParResult = 1 / ((1/1.44) + (1/(1j*(8/(3*math.pi))*ka[i])))
    zRadParApprox = np.append(zRadParApprox,zRadParResult)

    # Z Exact Fill-Up
    Zre = np.append(Zre, 1 - (spsp.jv(1,2*ka[i]))/ka[i])                          # Define Z real

    integrand = lambda x: math.sin(2*ka[i]*math.cos(x))*((math.sin(x))**2)        # Define integrand
    intgrxnResult = (scipy.integrate.quad(integrand,0,(math.pi/2)))               # Solve Integration

    Zim = np.append(Zim,(4/math.pi)*intgrxnResult[0])                             # Define Z imaginary

    zRadExact = np.append(zRadExact, Zre[i] + 1j*Zim[i])                          # Append each iteration's result to Zrad


plt.subplot(211)
#plt.loglog(ka, zRadLFapprox.real)
plt.loglog(ka, zRadParApprox.real)
#plt.loglog(ka, zRadExact.real)
plt.title('Real part of Normalized Z')
plt.grid(True)
plt.xlabel("ka (Normalized Frequency)")
plt.ylabel("Normalized Resistance")


plt.subplot(212)
plt.loglog(ka, zRadLFapprox.imag)
plt.loglog(ka, zRadParApprox.imag)
plt.loglog(ka, zRadExact.imag)
plt.title('Imaginary part of Normalized Z')
plt.grid(True)
plt.xlabel("ka (Normalized Frequency)")
plt.ylabel("Normalized Reactance")


plt.show()
