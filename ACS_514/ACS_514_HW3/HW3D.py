import numpy as np
import matplotlib.pyplot as plt
import math

md_Ad2 = 960
CdAd2 = 4.85E-14

CeoAd2_phi2 = 5.66E-13
phi = 1.7E-4

Ra1 = (10E7)     #/10
Ra2 = (36E9)     #10

Ca2 = 9E-13
Ad = 31E-6

m_arad = 73
R_arad = 19E6

# Ramp = 1E-8
# RampAcs = (Ramp*(Ad**2))/(phi**2)


Ze = Ra2

e_oc_p = np.array([])

freqs = np.logspace(np.log10(1),np.log10(100))
print(freqs)



for i in range(len(freqs)):
    w = freqs[i]*2*math.pi

    Za = (1/((1/R_arad)+(1/1j*w*m_arad)))+(1j*w*md_Ad2)+(1 / 1j * w * CdAd2)
    Zb = (1 / (1j * w * CeoAd2_phi2))
    # Zb = (1 / (1j * w * CeoAd2_phi2)+(1/RampAcs))
    Zc = (Ra1 - (1 / 1j * w * CeoAd2_phi2))
    Zd = (1 / (1j * w * Ca2))

    Zmtx = np.matrix([[Za+Zb+Zc+Zd,Zd],
                     [Zd,Ze+Zd]])

    U1_P = np.matrix([1,0])*np.linalg.inv(Zmtx)*np.matrix([[1],[1]])

    e_oc_p = np.append(e_oc_p,((Ad/phi)*U1_P*Zb))

plt.subplot(211)
plt.semilogx(freqs, abs(e_oc_p))
plt.title('Magnitude of eoc/p')
plt.grid(True)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Sensitivity (V/Pa)")

plt.subplot(212)
plt.semilogx(freqs, np.angle(e_oc_p)*(180/math.pi))
plt.title('Phase of eoc/p')
plt.grid(True)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (Degrees)")

plt.show()