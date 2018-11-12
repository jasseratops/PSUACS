import numpy as np
mgAr = 39.948E-3      # amu = g/mol
mgNe = 20.180E-3      # amu = g/mol

T = 293.15         # K
gamma = 1.667
R = 8.314          # J/mol.K
c = 350.0

mg = (T*R*gamma)/(c**2)

PerNe = (mg-mgAr)/(mgNe-mgAr)

print mg
print PerNe

check = np.sqrt(gamma*R*T/mg)
print check