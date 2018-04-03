import numpy as np
import cmath as cm


R1 = 1
R2 = 2
R3 = 3
R4 = 4
R5 = 5

Z = np.matrix('R1+R2 -R2 0; -R2 R2+R3+R4 R4; 0 R4 R4+R5')
Zinv = np.linalg.inv(Z)

IDvec = np.matrix('1; 0; 0')

print(Z)
print(Zinv)
print(IDvec)

V3oF1 = Zinv*IDvec

print('\n')
print(V3oF1)