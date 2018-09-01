import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

mode = 1

if mode == 0:                           #0 Impulse
    x_time = [1,0,0,0,0,0,0,0]

elif mode == 1:                         #1 Triangle
    x_time = [3,-3,3,-3,3,-3,3,-3,3,-3]

elif mode == 2:                         #2 Slow Triangle
    x_time = [3,0,-3,0,3,0,-3,0]

elif mode == 3:                         #3 Random
    x_time = [-2,4,6,-7,3,4,0,2]

elif mode == 4:                         #4 Step
    x_time = [1,1,1,1,1,1,1,1]

elif mode == 5:
    x_time = [3,1.5,0,-1.5,-3,-1.5,0,1.5,3,1.5,0,-1.5,-3,-1.5,0,1.5]

elif mode == 6:
    x_time = [0,3,0,-3,0,3,0,-3]


X_fft = np.fft.fft(x_time)

print X_fft

for i in range(len(X_fft)):
    #if X_fft[i].imag == 0:
    #    print i
    im = (X_fft[i]+X_fft[-i]).imag
    print str(i) + ": " + str(im)

plt.figure()
plt.plot(x_time)

plt.figure()
plt.plot(abs(X_fft))
plt.show()
