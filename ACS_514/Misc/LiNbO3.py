import numpy as np
from numpy.linalg import inv
import math


# PiezoElectric Compliance
s11 =  5.831E-12
s33 =  5.026E-12
s44 =  17.10E-12
s66 =  13.96E-12
s12 = -1.150E-12
s13 = -1.452E-12
s14 = -1.000E-12

se = np.matrix([[ s11, s12, s13, s14,    0,    0],       # PiezoElectric Compliance Matrix
                [ s12, s11, s13,-s14,    0,    0],
                [ s13, s13, s33,   0,    0,    0],
                [ s14,-s14,   0, s44,    0,    0],
                [   0,   0,   0,   0,  s44,2*s14],
                [   0,   0,   0,   0,2*s14,  s66]])


#Piezoelectric d coefficient
d15 =  69.2E-12
d31 = -0.85E-12
d22 =  20.8E-12
d33 =  6.00E-12

d = np.matrix([[   0,   0,   0,   0, d15,-2*d22],        # Piezoelectric d-coeff. Matrix
               [-d22, d22,   0, d15,   0,     0],
               [ d31, d31, d33,   0,   0,     0]])

dtr = np.matrix.transpose(d)

# Piezoelectric Permittivity
ezero= 8.854e-12

ept11= 85.2*ezero
ept33= 28.7*ezero

ept = np.matrix([[ept11,    0,    0],            # Piezoelectric Permittivity Matrix
                 [    0,ept11,    0],
                 [    0,    0,ept33]])

rho = 4640

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

'''
print("\nCeb = " + str(Ceb))
print("phi = " + str(phi))
print("Cmoc = " + str(Cmoc))
'''

cD33 = cd[2,2]
print("\ncD33 = " + str(cD33))

sD33 = 1/cD33
vD_33bar = 1/math.sqrt(rho*sD33)
print("\nsD33 = " + str(sD33))
print('vD_33bar = ' + str(vD_33bar))
