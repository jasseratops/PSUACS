import numpy as np
import matplotlib.pyplot as plt
import math

resFrq = 1.155E6                # define stimulus frequency as 1.155MHz
period = 1/resFrq               # define period of 1 cycle at 1.155MHz
numOfCyc = 41                   # given that 4 cycles should be, at most, 10%
totPer = period*numOfCyc        # calculate total period to end time domain linspace
ppCyc = 32                      # define resolution of each cycle
totPoints = numOfCyc*ppCyc      # define total points of the whole stimulus

nyqMult = 2                     # multiply
Fs = resFrq*nyqMult
Ts = 1/Fs

t = np.linspace(0, totPer ,totPoints)
stim = np.zeros(len(t))

for i in range(len(t)):
    x = 2*math.pi*(t[i]/period)

    if x < 2*math.pi*4:
        stim[i] = math.sin(x)
    else:
        stim[i] = 0

    #stim[i] = math.sin(x)

n = len(stim)
freq = np.fft.fftfreq(n,Ts/(nyqMult*numOfCyc))
freqDom = np.fft.fft(stim)

for i in range(len(stim)):
    if stim[i] == max(stim):
        print (freq[i])


print(freq)
print(totPer)

plt.subplot(211)
plt.plot(t,stim)

plt.subplot(212)
plt.semilogx(freq,freqDom.real)
plt.show()
