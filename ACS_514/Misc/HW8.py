import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math

radTodeg = 180/math.pi

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
#freq = np.fft.fftfreq(n,Ts/(nyqMult*numOfCyc))          # Part C
freq = np.fft.fftfreq(n,Ts/(ppCyc/nyqMult))          # Part C

FFTofStim = np.fft.fft(stim)

posSize = int(len(freq)/2)
posFreq = freq[:posSize]
posFFT = FFTofStim[:posSize]



##################
# bring circuit in
##################


# Piezoelectric Properties
tp = 1E-3                       # m
diam = 12E-3                    # m
Area = math.pi*((diam/2)**2)    # m^2
rho = 7500                      # kg/m^3
cD33 = 16E10                    # N/m^2
vDplate = math.sqrt((cD33/rho)) # m/s
Z0p = rho*vDplate*Area          # N.s/m
eS33 = 1.27E-8                  # F/m
Ceb = 1.43E-9                   # F
h33 = 18.3E8                    # V/m
phi = 2.62                      # A.s/m

# Water properties
rhoWater = 1000                 # kg/m^3
cW = 1500                       # m/s
R0 = rhoWater*cW*Area

# Structure properties

f0 = vDplate / (4 * tp)         # Hz
beta = Z0p / R0                 # Ratio

def v0_iZce(alpha):
    vm = cW
    v0_iZce = np.array([])      # initialize/clear v0_iZce upon every call for the function
    Z0m = alpha*R0
    tm = vm/(4*f0)

# Circuit Properties
    for i in range(len(posFreq)):
        w = posFreq[i]*2*math.pi
        if w == 0:              # to deal with zero division
            w += 1E-9
        Lp = tp
        kp = w/vDplate

        Zp1 = 1j*Z0p*math.tan((kp*Lp)/2)
        Zp2 = Z0p/(1j*math.sin(kp*Lp))

        Lm = tm
        km = w/vm

        Zm1 = 1j*Z0m*math.tan((km*Lm)/2)
        Zm2 = Z0m/(1j*math.sin(km*Lm))

        Zmtx = np.matrix([[Zm1+Zm2+R0, -Zm2],
                          [-Zm2, Zp1+Zp2+Zm1+Zm2]])

        v0_iZce = np.append(v0_iZce,(1/phi)*np.matrix([[1,0]])*np.linalg.inv(Zmtx)*np.matrix([[0],[1]]))

    return v0_iZce

def elByElMult(vecA,vecB):
    vecProd = np.zeros(len(vecA),dtype=np.complex_)     # initialize complex array
    for i in range(len(vecA)):
        lmnop = (vecA[i])*(vecB[i])
        vecProd[i] = lmnop
    return vecProd

def rebuildFFT(halfFFT):                                    # input one-sided FFT, to rebuild into two-sided FFT
    outFFT = np.zeros(2*len(halfFFT), dtype=np.complex_)    # init. output as complex array, with double size of input array.
    for i in range(len(halfFFT)):
        outFFT[i] = halfFFT[i]                              # build the pos. side of output to be equal to the input array
        outFFT[-i] = np.conj(halfFFT[i])                    # build the neg. side of output to be equal to conjugate of input array
    return outFFT


stimXdcr1 = elByElMult(rebuildFFT(posFFT),rebuildFFT(v0_iZce(1)))           # returns a double-sided FFT
stimXdcr2p5 = elByElMult(rebuildFFT(posFFT),rebuildFFT(v0_iZce(2.5)))       # returns a double-sided FFT

def indexMatchLow(matchVal,inputArray):
    for i in range(len(inputArray)):
        if inputArray[i] >= matchVal:
            return i

def indexMatchHigh(matchVal,inputArray):
    for i in range(len(inputArray)):
        if inputArray[i] <= matchVal:
            return i


pDifMinInd = indexMatchLow(875000,posFreq)
pDifMaxInd = indexMatchLow(1430000,posFreq)

phaseArray = np.unwrap(np.angle(v0_iZce(2.5)[:posSize]))

dPhase = phaseArray[pDifMaxInd]-phaseArray[pDifMinInd]
df = posFreq[pDifMaxInd]-posFreq[pDifMinInd]

taoPerT = -1*(stimFrq/(2*math.pi))*(dPhase/df)

print("tao per Tcycle = " + str(taoPerT))

plt.subplot(211).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.plot(t,stim,'g')
plt.xlim(0,totPer)
plt.title('CW-Pulse')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.grid(b=True, which='major', color='grey', linestyle='--')



plt.subplot(212).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.semilogy(posFreq,abs(posFFT),'g')
plt.xlim(posFreq[0],posFreq[-1])
plt.title('FFT of CW-pulse')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.grid(b=True, which='major', color='grey', linestyle='--')


# Plot Hydrophone Response and Phase

plt.figure()
plt.subplot(211).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))        #Formatting magic
plt.semilogy(posFreq,abs(v0_iZce(1)),'r', label = "alpha = 1")
plt.semilogy(posFreq,abs(v0_iZce(2.5)),'c',label = "alpha = 2.5")
plt.title('Magnitude of v0/iZce')
plt.ylabel("Response (in m/s/V)")
plt.xlabel("Frequency (in Hz)")
plt.xlim(posFreq[0],posFreq[-1])
plt.ylim(10E-8)
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()

plt.subplot(212).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))        #Formatting magic
plt.plot(posFreq,np.unwrap(np.angle(v0_iZce(1)))*radTodeg,'r', label = "alpha = 1")
plt.plot(posFreq,np.unwrap(np.angle(v0_iZce(2.5)))*radTodeg, 'c', label = "alpha = 2.5")
plt.title('Phase of v0/iZce')
plt.ylabel("Degrees")
plt.xlabel("Frequency (in Hz)")
plt.xlim(posFreq[0],posFreq[-1])
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()

#######################################
# PLOT TRANSDUCER RESPONSE TO STIMULUS
#######################################

plt.figure()
plt.subplot(211).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.semilogy(posFreq,abs(stimXdcr1[:posSize]),label="thru Hydrophone, alpha = 1")
plt.semilogy(posFreq,abs(stimXdcr2p5[:posSize]),label="thru Hydrophone, alpha = 2.5")
plt.title('FFT of Transducer Transform of Stimulus')
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.xlim(posFreq[0],posFreq[-1])
plt.ylim(1E-9)        # to chop off large dip at start; for readability
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()

plt.subplot(212).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.plot(posFreq,np.unwrap(np.angle(stimXdcr1[:posSize]))*radTodeg,label="thru Hydrophone, alpha = 1")
plt.plot(posFreq,np.unwrap(np.angle(stimXdcr2p5[:posSize]))*radTodeg,label="thru Hydrophone, alpha = 2.5")
plt.title('Phase of Transducer Transform of Stimulus')
plt.ylabel('Phase [deg]')
plt.xlabel('Frequency [Hz]')
plt.xlim(posFreq[0],posFreq[-1])
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()


#######################################
# ZOOM IN TO SMOOTH PHASE
#######################################

plt.figure()
plt.subplot(211).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))        #Formatting magic
plt.semilogy(posFreq[pDifMinInd:pDifMaxInd],abs(v0_iZce(2.5))[pDifMinInd:pDifMaxInd],'c',label = "alpha = 2.5")
plt.title('Magnitude of v0/iZce')
plt.ylabel("Response (in m/s/V)")
plt.xlabel("Frequency (in Hz)")
plt.xlim(posFreq[pDifMinInd],posFreq[pDifMaxInd])
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()

plt.subplot(212).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))        #Formatting magic
plt.plot(posFreq[pDifMinInd:pDifMaxInd],(np.unwrap(np.angle(v0_iZce(2.5)))[pDifMinInd:pDifMaxInd])*radTodeg, 'c', label = "alpha = 2.5")
plt.title('Phase of v0/iZce')
plt.ylabel("Degrees")
plt.xlabel("Frequency (in Hz)")
plt.xlim(posFreq[pDifMinInd],posFreq[pDifMaxInd])
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()

######




plt.figure()
plt.subplot(111).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.plot(t,np.fft.ifft(stimXdcr1), label="thru Hydrophone, alpha = 1")
plt.plot(t,np.fft.ifft(stimXdcr2p5), label="thru Hydrophone, alpha = 2.5")
plt.plot(t,stim/900,'g',label= 'CW-Pulse (Amplitude/900)')
plt.title('Stimulus Response')
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.xlim(t[0],t[-1])
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()


plt.figure()
plt.subplot(111).get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.2e'))
#plt.plot(t[:ppCyc*8],np.fft.ifft(stimXdcr1)[:ppCyc*8], label="thru Hydrophone, alpha = 1")
plt.plot(t[:ppCyc*8],(stim[:ppCyc*8])/900,'g',label= 'CW-Pulse (Amplitude/900)')
plt.plot(t[:ppCyc*8],np.fft.ifft(stimXdcr2p5)[:ppCyc*8],'darkorange', label="thru Hydrophone, alpha = 2.5")
plt.title('Stimulus Response')
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.xlim(t[0],t[ppCyc*8])
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()

plt.show()