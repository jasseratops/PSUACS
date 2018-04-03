import matplotlib.pyplot as plt
import math
import numpy as np

freq = np.linspace(0,127,128)

logar = np.zeros(np.shape(freq))

def logarFunc(base):
    logar = np.zeros(np.shape(freq))
    print("base = " + str(base))
    print("log(127," +str(base) + ") = " +str(math.log(127,base)))
    for i in range(len(freq)):
        logar[i] = math.log(i+1,base)*127/math.log(127,base)
    return logar


plt.subplot(111)

for x in range(5):
    plt.plot(freq,logarFunc(x+2), label = "base = " + str(x+2))

plt.legend()
plt.show()