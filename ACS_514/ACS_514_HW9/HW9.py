import numpy as np
import matplotlib.pyplot as plt
import math


def Parts_AthruD():
    f = np.fromfile(file="SpeakerZ.txt", sep = " ")
    noOfRows = len(f)/3

    freqs = np.zeros(noOfRows)
    reZ = np.zeros(noOfRows)
    imZ = np.zeros(noOfRows)
    difReIm = np.zeros(noOfRows)
    y = 0


    for i in range(len(f)):
        x = i%3
        if x == 0:
            freqs[y] = f[i]
        elif x == 1:
            reZ[y] = f[i]
        else:
            imZ[y] = f[i]
            y += 1

    compZ = reZ + 1j*imZ

    f0ind = np.argmax(reZ)
    f0 = freqs[f0ind]
    print(f0)

    f2ind = np.argmin(imZ)
    f2 = freqs[f2ind]
    print(f2)

    f1ind = np.argmax(imZ[:f2ind])
    f1 = freqs[f1ind]
    print(f1)

    zeroCrossInd = np.argmin(abs(imZ[f1ind:f2ind]))+f1ind
    zeroCrossF = freqs[zeroCrossInd]
    print(zeroCrossF)

    return(freqs,reZ,imZ,f0ind,f1ind,f2ind,zeroCrossF)

def printNplot(freqs,reZ,imZ):
    plt.plot(reZ,imZ)               # plot of Im[Z] vs. Re[Z]
    plt.xlim(6,17)

    plt.figure()
    plt.subplot(211)
    plt.plot(freqs,reZ)             # Plot of Re[Z] vs. Frequency
    plt.xlim(freqs[0],freqs[-1])

    plt.subplot(212)
    plt.plot(freqs,imZ)
    plt.xlim(freqs[0],freqs[-1])

    plt.show()

def Part_E():
    addedMass = [0,11.25,23.5,35.7,47.1]                # init. added mass array
    aMresFrq = [95.4,67.5,55.8,49.5,45.0]               # init. different resonance freq. array for added masses

    Cm = np.zeros(len(addedMass)-1)                     # init. calc. mech. compliance array

    f0 = aMresFrq[0]                                    # define resonant frequency

    for i in range(1,len(addedMass)):                   # calc. Cm for each freq./mass combination
        f = aMresFrq[i]
        delM = addedMass[i]
        Cm[i-1] = ((1/(f**2))-(1/(f0**2)))*(1/delM)*(1/(4*(math.pi**2)))

    avgCm = np.mean(Cm)                                 # find average calculated Cm

    m = ((1/(f0)**2))*(1/(4*(math.pi**2)))*(1/avgCm)    # calc. Mechanical mass

    print('Mechanical Compliance: ' + str(avgCm))
    print('Mechanical Mass: ' + str(m))

    return(m,avgCm)

def Part_F(freqs,reZ,imZ,f1ind,f2ind):
    Zelectr = np.zeros(len(freqs),dtype=complex)

    m = 0.015                                           # kg
    f0 = 95.4                                           # Hz
    f1 = freqs[f1ind]
    f2 = freqs[f2ind]

    imZ_f1 = imZ[f1ind]
    imZ_f2 = imZ[f2ind]

    Q = f0 / (f2 - f1)
    Cm = f0 / (4 * m * (math.pi ** 2))
    Rm = (2 * math.pi) * f0 * m
    two_a = abs(imZ_f2 - imZ_f1)

    Bl = math.sqrt((two_a*Rm))
    R0e = np.max(reZ)-two_a

    for i in range(len(freqs)):
        omega = freqs[i]/(f0)

        Zelectr[i] = R0e + (((1j*omega)/Q)/(1-(omega**2)+((1j*omega)/Q)))*((Bl**2)/Rm)

    printNplot(freqs,Zelectr.real,Zelectr.imag)

def main():
    AthruDarray = Parts_AthruD()

    freqs = AthruDarray[0]
    reZ = AthruDarray[1]
    imZ = AthruDarray[2]
    f1ind = AthruDarray[4]
    f2ind = AthruDarray[5]

    printNplot(freqs,reZ,imZ)
    Part_E()
    Part_F(freqs,reZ,imZ,f1ind,f2ind)


if __name__ == "__main__":
    main()