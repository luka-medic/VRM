import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
a = [[1, 2],
     [1, 4]]
b = [3, 4]

a = np.array(a)
b = np.array(b)

# print(b[:, np.newaxis])

c = np.array([1, 2, 3, 4])

# print(c.reshape((2, 2), order='F'))

c = np.array([i for i in range(16)])
c = c.reshape((4, 4))

print(c)

c = c.reshape((2, 2, 4))

print(c)

print(184236/1048576)

n = 20
print(2**n, comb(n, n//2))

ns = range(4, 22, 2)
ratios = []

np.load('ground_states')

for n in ns:
     ratios.append(comb(n, n//2)/2**n)

plt.plot(ns, ratios)
plt.show()
