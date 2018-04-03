import numpy as np
from numpy import sqrt, sin, pi, arctan,rad2deg

m = 1
n = 3

T=300
rho=2.5
Lx=0.60
Lz=0.20

f = (1/(6*Lz))*(sqrt(T/rho))*sqrt((n**2)+(9*(m**2)))

print f

y = 2.00E-2
C = sin(((2*pi)/Lx)*0.15)*sin(((2*pi)/Lz)*0.05)
A = y/C
print A

yxz= A*sin(((2*pi)/Lx)*0.20)*sin(((2*pi)/Lz)*0.12)
print yxz

""""""

F = 2
m=0.5
s=100
R=1.4
omega = 31.4

u = F/(R+(1j*((omega*m)+(s/omega))))
MAXu = abs(u)
print MAXu

Xm = (omega*m)+(s/omega)
theta = rad2deg(arctan(Xm/R))
print "Theta: ", theta

''''''

F0= 100.00
omega= 5000.00
Y =10.40E10
r = 0.015
S = pi*(r**2)
c = 3500.00
k=omega/c
print "k: ",k
print "S: ",S

xi = F0/(Y*S*k)
print "xi: ", xi
