import numpy as np
import matplotlib.pyplot as plt
import math


def Parts_AthruD():
    print('Parts A, B, C, and D ' + 11*'-')
    f = np.fromfile(file="SpeakerZ.txt", sep = " ")
    noOfRows = int(len(f)/3)

    freqs = np.zeros(noOfRows)
    reZ = np.zeros(noOfRows)
    imZ = np.zeros(noOfRows)

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

    f0ind = np.argmax(reZ)
    f0 = freqs[f0ind]
    print("f0 = " + str(f0) + "Hz")
    print("Re[Z](f0) = " + str(reZ[f0ind]) + " Ohms")

    f2ind = np.argmin(imZ)
    f2 = freqs[f2ind]
    print("\nf2 = " + str(f2) + " Hz")
    print("Im[Z](f2) = " + str(imZ[f2ind]) + " Ohms")

    f1ind = np.argmax(imZ[:f2ind])
    f1 = freqs[f1ind]
    print("\nf1 = " +str(f1) + " Hz")
    print("Im[Z](f1) = " + str(imZ[f1ind]) + " Ohms")


    zeroCrossInd = np.argmin(abs(imZ[f1ind:f2ind]))+f1ind
    zeroCrossF = freqs[zeroCrossInd]
    print("Imag. Imp. Zero-Crossing Freq. = " + str(zeroCrossF) + " Hz")

    return(freqs,reZ,imZ,f0ind,f1ind,f2ind,zeroCrossF)

def plotterFunc(freqs,reZ,imZ, Zelectr):
    #lns.set_linewidth(10)

    plt.figure()
    plt.plot(reZ,imZ)
    plt.xlim(6,15)
    plt.ylim(-3,6)
    plt.title('Nyquist Plot of the Given Impedance')
    plt.xlabel('Re[Z]')
    plt.ylabel('Im[Z]')
    plt.grid(b=True, which='major', color='grey', linestyle='--')

    plt.figure()
    plt.subplot(211)
    plt.semilogx(freqs,reZ)
    plt.xlim(freqs[0],freqs[-1])
    plt.title('Given Re[Z] vs. Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Re[Z] (Ohms)')
    plt.grid(b=True, which='major', color='grey', linestyle='--')

    plt.subplot(212)
    plt.semilogx(freqs,imZ)
    plt.xlim(freqs[0],freqs[-1])
    plt.title('Given Im[Z] vs. Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Im[Z] (Ohms)')
    plt.grid(b=True, which='major', color='grey', linestyle='--')

    plt.figure()
    plt.plot(reZ,imZ,label= "Given Z")
    plt.plot(Zelectr.real,Zelectr.imag, label= "Calc. Z")
    plt.xlim(6,15)
    plt.ylim(-3,6)
    plt.title('Nyquist Plots (Given vs. Calculated)')
    plt.xlabel('Re[Z] (Ohms)')
    plt.ylabel('Im[Z] (Ohms)')
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.legend()

    plt.figure()
    plt.subplot(211)
    plt.semilogx(freqs, reZ, label= "Given Real Z")
    plt.semilogx(freqs, Zelectr.real, label= "Calc. Real Z")
    plt.xlim(freqs[0],freqs[-1])
    plt.title('Given Re[Z] vs. Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Re[Z] (Ohms)')
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.legend()

    plt.subplot(212)
    plt.semilogx(freqs, imZ, label= "Given Imag. Z")
    plt.semilogx(freqs, Zelectr.imag, label= "Calc. Imag. Z")
    plt.xlim(freqs[0],freqs[-1])
    plt.title('Given Im[Z] vs. Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Im[Z] (Ohms)')
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.legend()

    plt.show()

def Part_E():
    print("\nPart E " + 10*'-')

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

    print('Cm = ' + str(avgCm) + " m/N")
    print('m = ' + str(m) + ' kg')

    return(m,avgCm)

def Part_F(freqs,reZ,imZ,f1ind,f2ind):
    print("\nPart F " + 10*'-')

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
    two_a = abs(imZ_f2-imZ_f1)

    Bl = math.sqrt((two_a*Rm))
    R0e = np.max(reZ)-two_a

    print("Q = " + str(Q))
    print("Cm = " + str(Cm) + " m/N")
    print("Rm = " + str(Rm) + " Ohms")
    print("2a = " + str(two_a) + " Ohms")
    print("Bl = " + str(Bl) + " Ohms")
    print("R0e = " + str(R0e) + " Ohms")

    for i in range(len(freqs)):
        omega = freqs[i]/(f0)
        Zelectr[i] = R0e + (((1j*omega)/Q)/(1-(omega**2)+((1j*omega)/Q)))*((Bl**2)/Rm)

    return(Zelectr)

def main():
    AthruDarray = Parts_AthruD()

    freqs = AthruDarray[0]
    reZ = AthruDarray[1]
    imZ = AthruDarray[2]
    f1ind = AthruDarray[4]
    f2ind = AthruDarray[5]

    Part_E()
    Zelectr = Part_F(freqs,reZ,imZ,f1ind,f2ind)
    plotterFunc(freqs,reZ,imZ,Zelectr)

if __name__ == "__main__":
    main()