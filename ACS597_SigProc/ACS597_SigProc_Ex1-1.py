import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

timeVec = [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0]

delT = 0.01
fs = 1.0/delT

linSpec = np.fft.fft(timeVec)*delT

print linSpec

plt.figure()
plt.plot(abs(linSpec))
plt.show()