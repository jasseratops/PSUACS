import numpy as np
import matplotlib.pyplot as plt
import math

freqs = np.zeros(81)                        # Initialize arrays
reZ = np.zeros(81)
imZ = np.zeros(81)
line = np.zeros(81)
compZ = np.zeros(81,dtype=complex)

f = np.fromfile(file="Piezo_Impedance.txt", sep = ' ')  # Import text file data in array f
y = 0                                       # set sorting index 0

for i in range(len(f)):                     # Re-sort array into separate vectors for...
    x = i%3
    if x == 0:
        freqs[y] = f[i]                     # ... frequency
    elif x == 1:
        reZ[y] = f[i]                       # ... real Z
    else:
        imZ[y] = f[i]                       # ... and imaginary Z.
        y +=1

compZ = reZ + 1j*imZ                        # Calculate Complex Impedance.
compY = 1/compZ                             # Calculate Complex Admittance.

reY = compY.real                            # Separate out real and ...
imY = compY.imag                            # ... imag parts of Admittance.

omega = 2*math.pi*freqs                     # Generate angular frequency vector.


# Generate linear equation for the increasing offset in the imag Y vector
# This is to eventually calculate Ce
slope = (imY[-1] - imY[0])/(freqs[-1] - freqs[0])
offset = imY[0] - freqs[0]*slope
line = (slope*freqs) + offset

adjImY = imY-line                           # Calculate Adjusted Admittance vector

f1ind = np.argmax(adjImY)                   # Determine f1
f1 = freqs[f1ind]

f2ind = np.argmin(adjImY)                   # Determine f2
f2 = freqs[f2ind]

f0ind = np.argmax(reY)                      # Determine f0
f0 = freqs[f0ind]

Qm = f0/(f2-f1)                             # calculate Qm

CeInd = 65                                  # Index used to determine Ce. 65 was chosen because it gave the closest estimate.

Re = 1/min(reY)
Ce = line[CeInd]/omega[CeInd]               # arbit. pick a point in the offset line to calc. electrical cap
Rme = 1/(max(reY)-(1/Re))
Mme = (Qm*Rme)/omega[f0ind]
Cme = 1/(Mme*((omega[f0ind])**2))

print("\n")
print("f0: " + str(f0))
print("Re: " + str(Re))
print("Ce: " + str(Ce))
print("Rme: " + str(Rme))
print("Mme: " + str(Mme))
print("Cme: " + str(Cme))
print("Qm: " + str(Qm))

Yme = (1j*omega*Cme)/(1-((omega**2)*Mme*Cme) + (1j*omega*Rme*Cme))  # Calculate mechanical Admittance
Ytot = ((1j*omega*Cme)+(1/Re) + Yme)                                # Calculate total admittance
YtotAdj = (Ytot.real) + 1j*(Ytot.imag+(omega*Ce))                   # Adjust total admittance for omega*Ce

calcZ = 1/YtotAdj                                                   # Calculate circuit impedance from estimated admittance

# Initial Admittance Plots
plt.subplot(211)
plt.title("Re[Admittance]")
plt.plot(freqs,reY, label = "Real Y")
plt.plot(freqs,reY-(1/Re), label = "Real Y - (1/Re)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Re[Y] (S)")
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.xlim(freqs[0],freqs[-1])
plt.legend()

plt.subplot(212)
plt.title("Im[Admittance]")
plt.plot(freqs,imY, label = "Imag. Y")
plt.plot(freqs,line, label = "omega*Ce")
plt.plot(freqs,adjImY, label = "Imag. Y - omega*Ce")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Im[Y] (j*S)")
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.xlim(freqs[0],freqs[-1])
plt.legend()



# Admittance Plots, Given vs. Calc
plt.figure()
plt.subplot(211)
plt.title("Re[Admittance], Given vs. Calculated")
plt.plot(freqs,reY, label = "Real Y")
plt.plot(freqs,YtotAdj.real, label = "Calculated Real Y")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Re[Y] (S)")
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.xlim(freqs[0],freqs[-1])
plt.legend()

plt.subplot(212)
plt.title("Im[Admittance], Given vs. Calculated")
plt.plot(freqs,imY, label = "Imag. Y")
plt.plot(freqs,YtotAdj.imag, label = "Calc. Imag. Y")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Im[Y] (j*S)")
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.xlim(freqs[0],freqs[-1])
plt.legend()

# Impedance Plots, Given vs. Calculated
plt.figure()
plt.subplot(211)
plt.title("Re[Impedance], Given vs. Calculated")
plt.plot(freqs,reZ, label = "Real Z")
plt.plot(freqs,calcZ.real, label="Calc. Real Z")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Re[Z] (Ohms)")
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.xlim(freqs[0],freqs[-1])
plt.legend()


plt.subplot(212)
plt.title("Im[Impedance], Given vs. Calculated")
plt.plot(freqs,imZ, label = "Imag. Z")
plt.plot(freqs,calcZ.imag, label = "Calc. Imag. Z")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Im[Z] (j*Ohms)")
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.xlim(freqs[0],freqs[-1])
plt.legend()

plt.show()