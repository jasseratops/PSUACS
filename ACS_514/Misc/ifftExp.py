import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

a = 0

stimFrq = 1.155E6               # define stimulus frequency as 1.155MHz
period = 1/stimFrq              # define period of 1 cycle at stimulus frequency
numOfCyc = 41                   # given that 4 cycles should be, at most, 10%
totPer = period*numOfCyc        # calculate total period to end time domain linspace
ppCyc = 32                      # define resolution of each cycle
totPoints = numOfCyc*ppCyc      # define total points of the whole stimulus

nyqMult = 2                     # multiply
Fs = stimFrq*nyqMult
Ts = 1/Fs

t = np.linspace(0, totPer ,totPoints)
stim = np.zeros(len(t))

for i in range(len(t)):
    x = 2*math.pi*(t[i]/period)

    if x < 2*math.pi*4:
        stim[i] = math.sin(x)
    else:
        stim[i] = 0

    #stim[i] = math.sin(x)                              # Creates a continuous stimulus, vs CW

n = len(stim)
freq = np.fft.fftfreq(n,Ts/(ppCyc/nyqMult))          # Part C

FFTofStim = np.fft.fft(stim)

posSize = int(len(freq)/2)
posFreq = freq[:posSize]
posFFT = FFTofStim[:posSize]

indexMax = np.argmax(abs(posFFT))

print(posFreq[0])

print(indexMax)
maxFrq = posFreq[indexMax]
print(maxFrq)
missingConst = 1/(max(abs(posFFT)))     #knowing that max amplitude = 1 in the time domain. This should be reflected in freq domain
actualFFT = posFFT*missingConst

def rebuildFFT(halfFFT):
    outFFT = np.zeros(2*len(halfFFT), dtype=np.complex_)
    for i in range(len(halfFFT)):
        outFFT[i] = halfFFT[i]
        outFFT[-i] = np.conj(halfFFT[i])
    return outFFT


inverseFFT = np.fft.ifft(rebuildFFT(posFFT))

rebuiltFFT = rebuildFFT(posFFT)


for i in range(len(inverseFFT)):
    print("i: " + str(i))
    print(rebuiltFFT[i] - FFTofStim[i])


#print(rebuiltFFT)

for i in range(len(FFTofStim)):
    print("i: " + str(i))
    print(FFTofStim[i])
    print(FFTofStim[-i])
    print("\n")

plt.subplot(111)
plt.plot(t,stim)
plt.plot(t,inverseFFT)
plt.plot(t,np.fft.ifft(FFTofStim))
plt.figure()

plt.subplot(111)
plt.plot(freq,FFTofStim)
plt.plot(freq,rebuildFFT(posFFT))
plt.figure()

plt.subplot(111)
plt.plot(freq,np.angle(FFTofStim))
plt.plot(freq,np.angle(rebuildFFT(posFFT)))

plt.show()

