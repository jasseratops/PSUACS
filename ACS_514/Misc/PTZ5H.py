import numpy as np
from numpy.linalg import inv


# PiezoElectric Compliance
s11=16.5e-12; s12=-4.78e-12; s13=-8.45e-12;
s33=20.7e-12; s44=43.5e-12; s66=2*(s11-s12);

se = np.matrix([[s11,s12,s13,  0,  0,  0],       # PiezoElectric Compliance Matrix
                [s12,s11,s13,  0,  0,  0],
                [s13,s13,s33,  0,  0,  0],
                [  0,  0,  0,s44,  0,  0],
                [  0,  0,  0,  0,s44,  0],
                [  0,  0,  0,  0,  0,s66]])


#Piezoelectric d coefficient
d31=-274e-12; d33=593e-12; d15=741e-12;

d = np.matrix([[  0,  0,  0,  0,d15,  0],        # Piezoelectric d-coeff. Matrix
               [  0,  0,  0,d15,  0,  0],
               [d31,d31,d33,  0,  0,  0]])

dtr = np.matrix.transpose(d)

# Piezoelectric Permittivity
ezero=8.854e-12;
ept11=3130*ezero; ept33=3400*ezero;

ept = np.matrix([[ept11,    0,    0],            # Piezoelectric Permittivity Matrix
                 [    0,ept11,    0],
                 [    0,    0,ept33]])

rho = 7500

upperMTX = np.concatenate((se,dtr),axis=1)
lowerMTX = np.concatenate((d,ept),axis=1)


composite = np.concatenate((upperMTX,lowerMTX), axis=0)
compINV = inv(composite)

# initialize sub-matrices with data types set to float
cd     = np.zeros((6,6),dtype=float)
negh   = np.zeros((3,6),dtype=float)
betaS  = np.zeros((3,3),dtype=float)

# Formatting magic
np.set_printoptions(formatter={'float': lambda x: format(x, '6.3E')})

# grab sub-matrices from 9x9 inverted composite matrix

cd     = compINV[:6,:6]
negh   = compINV[6:,:6]
betaS  = compINV[6:,6:]
h = -1*negh

print("\ncD (in N/m^2): ")
print(cd)
print("\nh (in N/C): ")
print(h)
print("\nbetaS (in m/F): ")
print(betaS)


betaS11 = betaS[0,0]
h15 = h[0,4]
cD44 = cd[4,4]

w = 1E-2
l = 1E-2
t = 1E-3


Ceb = (w*l)/(betaS11*t)
phi = -h15*Ceb
Cmoc = t/(cD44*w*l)

print("\nCeb = " + str(Ceb))
print("phi = " + str(phi))
print("Cmoc = " + str(Cmoc))
