import numpy as np
from numpy import pi
from datetime import datetime, timedelta

diameter = 0.07     # m
r = diameter/2.0    # m
T0 = 20.0           # C
T1 = 100.0          # C
dT = T1-T0          # K
V = 0.250           # L
rho = 1.0           # 1kg/L
m = rho*V           # kg
SPL = 142.2         # dBSPL
Iref = 1.0E-12      # W/m^2
Pref = 20E-6        # Pa
C = 4.186E3         # J/kg
rho0 = 1.21         # kg/m^3
c0 = 343.0          # m/s

prms = (Pref*(10**(SPL/20)))#/np.sqrt(2)

print "Prms: " + str(prms)

Q = m*C*dT

print "Q: " + str(Q)

IL = SPL - 0.16
I = Iref*(10**(IL/10.0))
I2 = (prms**2)/(rho0*c0)

print "IL: " + str(IL)
print "I: " + str(I)
print "I2: " + str(I2)

S = pi*(r**2)

W = I2*S

print "S: " + str(S)
print "W: " + str(W)

timeS = Q/W

print "Time [s]" + str(timeS)

sec = timedelta(seconds=int(timeS))
d = datetime(1,1,1) + sec

print("DAYS:HOURS:MIN:SEC")
print("%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second))