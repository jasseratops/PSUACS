import numpy as np

SPL = 185.0             # dB
Pref = 1.0E-6           # Pa
f = 5.0E3               # Hz

rhoW = 1026.0           # kg/m^3
cW = 1500.0             # m/s

# i) Find Prms

Ppk = Pref*(10**(SPL/20))
Prms = Ppk/np.sqrt(2)

print "Ppk: " + str(Ppk)
print "Prms: " + str(Prms)

# ii) Find urms

u = (1.0/(rhoW*cW))*Prms

print "u(rms): " + str(u)

# iii) find xirms

xi = (1.0/(rhoW*cW))*(1.0/(2.0*np.pi*f))*Prms

print "xi(rms): " + str(xi)

# iv) find rhorms

rho = Prms/(cW**2)
print "rho(rms): " + str(rho)
print 10*"-"
# v) Intensity

I = (Prms**2)/(rhoW*cW)
print "I(rms): " + str(I)

# vi) find IL

Iref = (Pref**2)/(rhoW*cW)

IL = 10.0*np.log10(I/Iref)

print "Iref: " + str(Iref)
print "IL: " + str(IL)

print SPL-IL