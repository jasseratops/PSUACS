import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

n = 5
omega = np.linspace(-pi,pi,200)

HLPF = np.sinc(n*omega)
HLPFsq = np.power(HLPF,2)

abSum = np.sum(HLPF)
sqSum = np.sum(HLPFsq)


plt.figure()
plt.plot(omega,HLPF)
plt.plot(omega,HLPFsq)
plt.show()