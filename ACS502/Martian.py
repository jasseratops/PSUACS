import numpy as np

P = 600.0                 # Pa
T = 210.0                 # K
R = 8.314                 # J/mol.K


perCO2 = 95.97E-2
perAr = 1.93E-2
perN2 = 1.89E-2
perO2 = 0.146E-2
perCO = 0.0557E-2

mgCO2 = 44.01E-3
mgAR = 39.95E-3
mgN2 = 28.01E-3
mgO2 = 32.00E-3
mgCO = 28.01E-3

gamCO2 = 1.333
gamAR = 1.667
gamN2 = 1.4
gamO2 = 1.4
gamCO = 1.4

gamma = perCO2*gamCO2 + perAr*gamAR + perN2*gamN2\
        + perO2*gamO2 + perCO*gamCO

mg = perCO2*mgCO2 + perAr*mgAR + perN2*mgN2\
        + perO2*mgO2 + perCO*mgCO

rho = mg*(P/(R*T))

c = np.sqrt((gamma*R*T)/mg)

cEarth = 343.0
lamLow = cEarth/20.0
lamHigh = cEarth/(20.0E3)

print lamLow
print lamHigh

frqLow = c/lamLow
frqHigh = c/lamHigh

print "mg: " + str(mg)
print "gamma: " + str(gamma)
print "rho: " + str(rho)
print "Cmars: " + str(c)

print str(frqLow) + " - " + str(frqHigh)