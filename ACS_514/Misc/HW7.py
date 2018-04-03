import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib

freqs = np.linspace(100E3,2.2E6,1000)

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

def v0_iZce(alpha,vm):
    v0_iZce = np.array([])      # initialize/clear v0_iZce upon every call for the function
    Z0m = alpha*R0
    tm = vm/(4*f0)

# Circuit Properties
    for i in range(len(freqs)):
        w = freqs[i]*2*math.pi
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

def findF1(curve):
    F1 = 0
    PEAK = 0.707 * max(curve)
    for x in range((len(curve))):
        if curve[x] >= PEAK:
            F1 = freqs[x]
            break
    return F1

def findF2(curve):
    F2 = 0
    PEAK = 0.707 * max(curve)

    for x in range((len(curve))):
        if curve[len(curve)-1-x] >= PEAK:
            F2 = freqs[len(curve)-1-x]
            break

    return F2

def findBW(curve):
    ThreedB1 = findF1(curve)
    ThreedB2 = findF2(curve)
    BW = ThreedB2 - ThreedB1
    BWarray = [ThreedB1,ThreedB2,BW]
    return BWarray



E1array = findBW(abs(v0_iZce(1,cW)))
E4array = findBW(abs(v0_iZce(2.4,cW)))

print("E.1:")
print("F1 = " + str(E1array[0]) + ", F2 = " + str(E1array[1]))
print("BW = " + str(E1array[2]))

print("\nE.4:")
print("F1 = " + str(E4array[0]) + ", F2 = " + str(E4array[1]))
print("BW = " + str(E4array[2]))


plt.subplot(211).get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))        #Formatting magic

plt.semilogy(freqs,abs(v0_iZce(1,cW)), label = "alpha = 1")
# plt.semilogy(freqs,abs(v0_iZce(1,3000)))
plt.semilogy(freqs,abs(v0_iZce(math.sqrt(beta),cW)),label = "alpha = sqrt(beta)")
plt.semilogy(freqs,abs(v0_iZce(2.4,cW)), label = "alpha = 2.4")

plt.title('Magnitude of v0/iZce')
plt.ylabel("Response (in m/s/V)")
plt.xlabel("Frequency (in Hz)")
plt.xlim(freqs[0],freqs[-1])
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()

plt.subplot(212).get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))        #Formatting magic

plt.plot(freqs,np.angle(v0_iZce(1,cW))*180/math.pi, label = "alpha = 1")
# plt.plot(freqs,np.angle(v0_iZce(1,3000))180/math.pi)
plt.plot(freqs,np.angle(v0_iZce(math.sqrt(beta),cW))*180/math.pi, label = "alpha = sqrt(beta)")
plt.plot(freqs,np.angle(v0_iZce(2.4,cW))*180/math.pi, label = "alpha = 2.4")

plt.title('Phase of v0/iZce')
plt.ylabel("Degrees")
plt.xlabel("Frequency (in Hz)")
plt.ylim(-200,200)
plt.xlim(freqs[0],freqs[-1])
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend()

plt.show()