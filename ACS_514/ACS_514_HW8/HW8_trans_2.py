import matplotlib.pyplot as plt
import numpy as np
import math
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

resFrq = 1.155E6
per = 1/resFrq
numOfCyc = 41
totPer = per*numOfCyc
ppCyc = 32
totPoints = numOfCyc*ppCyc
fourCycP = ppCyc*4
nyqMult = 2.1
fourCycX = math.pi*4*2

Fs = resFrq*nyqMult  # sampling rate
Ts = totPer/Fs # sampling interval
t = np.arange(0,totPer,Ts) # time vector
print(len(t))

ff = resFrq;   # frequency of the signal
y = np.zeros(len(t))

for i in range(len(t)):
    x = 2*np.pi*ff*t
    if x[i] <= fourCycX:
        y[i] = np.sin(2*np.pi*ff*t[i])
    else:
        y[i] = 0

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = (k/T)/nyqMult # two sides frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n)]

for i in range(n):
    if Y[i] == max(Y):
        print (frq[i])

print("about to plot")

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot((frq/2.5),abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
plt.show()