import matplotlib.pyplot as plt
import math
import numpy as np

steps = np.linspace(0,127,128)
depth = 5

linear = np.zeros(np.shape(steps))
expon = np.zeros(np.shape(steps))
logar = np.zeros(np.shape(steps))

for i in range(len(steps)):
    linear[i] = i
    expon[i] = ((i)**depth)/(127**(depth-1))
    logar[i] = math.log(i+1)*127/math.log(127)

print(np.shape(linear))
print(np.shape(steps))

plt.subplot(111)
plt.plot(steps,linear,label ="linear")
plt.plot(steps,expon,label = "exponential")
plt.plot(steps,logar,label = "logarithmic")
plt.legend()
plt.show()