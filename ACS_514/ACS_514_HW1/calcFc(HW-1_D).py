import numpy as np
import math

M1 = 0.01

C2 = 0.01
C3 = 0.01

R1 = 0.01
R2 = 0.01
R3 = 0.01

F1 = 1
F3 = 0

frq = 100

w =frq*2*math.pi

print("w: " + str(w))

Fmtx = np.matrix([[1],
                  [0],
                  [0]])

Zmtx = np.matrix([[(1j*w*M1)+R1,-R1,0],
                  [-R1,R1+R2+(1/(1j*w*C2)),-R2],
                  [0,-R2,R2+R3+(1/(1j*w*C3))]])

ZmtxInv = np.linalg.inv(Zmtx)

VsOverF1 = ZmtxInv*Fmtx

print("\nF Vector: " + str(Fmtx))
print("\nZ matrix: " + str(Zmtx))
print("\nInverted Z: " + str(ZmtxInv))
print("\nVelocity Vector/F1: " + str(VsOverF1))

FC2 = VsOverF1[1]/(1j*w*C2)

print("FC2 = " + str(FC2))
