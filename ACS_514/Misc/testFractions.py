import matplotlib.pyplot as plt
import numpy as np
x = 5
y = 8
z = 6
a = 9

t = np.linspace (1,100,100)

print(t)

first = np.zeros(np.shape(t))
second = np.zeros(np.shape(t))

for i in range(len(t)):
    first[i] = x/y
    second[i] = z/a
    x += 1
    y += 1
    z += 1
    a += 1

print(first)
print(second)

plt.plot(t,first, label="first")
plt.plot(t,second, label="second")
plt.legend()
plt.show()