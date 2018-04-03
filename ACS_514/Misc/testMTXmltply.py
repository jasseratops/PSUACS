import numpy as np

x = 5
a = np.matrix([[0,x],
               [1/x,0]])
b = a

c = a*b

print(c)
