import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
# import scipy.integrate as integrate
from scipy.optimize import curve_fit
import time
# import scipy.fftpack as fft
import matplotlib.gridspec as gridspec
import math
from mpl_toolkits.mplot3d import axes3d
# import scipy.sparse as sparse
# from matplotlib.animation import FuncAnimation, ArtistAnimation
# import matplotlib.animation as animation
# from scipy.integrate import odeint, solve_ivp, RK45
# from scipy.signal import correlate
import matplotlib.gridspec as gridspec
# import numba as nb

plt.rcParams['animation.ffmpeg_path'] = 'C:/FFmpeg/bin/ffmpeg'
plt.rc('text', usetex=0)
plt.rc('font', size=16, family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('legend', fontsize='small')
plt.rc('figure', figsize=(8, 6))
plt.rc('lines', linewidth=1.0)

h2 = [[1, 0, 0, 0],
      [0,-1, 2, 0],
      [0, 2,-1, 0],
      [0, 0, 0, 1]]
h2 = np.array(h2)

sx = [[0, 1], [1, 0]]
sy = [[0,-1j], [1j, 0]]
sz = [[1, 0], [0,-1]]
sx, sy, sz = np.array(sx), np.array(sy), np.array(sz)


def reshape_index(num, sa, sb):
    string = bin(num)[2:]
    string = '0'*(n-len(string))+string
    bin_array = np.array([int(x) for x in string])[::-1]

    i = sum([c*2**k for k, c in enumerate(bin_array[sa])])
    j = sum([c*2**k for k, c in enumerate(bin_array[sb])])

    return (i, j)


# fig = plt.figure()

periodic = True
n = 5

# Construct Hamiltonian
H = np.zeros((2**n, 2**n))
for k in range(n-1):
    H += np.kron(np.identity(2**k), np.kron(h2, np.identity(2**(n-k-2))))

# Periodic boundary condition
if periodic:
    for s in (sx, sy, sz):
        H += np.real(np.kron(s, np.kron(np.identity(2**(n-2)), s)))

# Get ground state
eig, vec = np.linalg.eigh(H)
psi = vec[:, 0]

psi = np.random.standard_normal(2**n)

print(psi, sum(psi**2))

# Reshape psi -> Psi
sa, sb = np.array([k for k in range(1)]), np.array([k for k in range(1, n)])
a, b = 2**sa.size, 2**sb.size
Psi = np.zeros((a, b))

# for k, m in enumerate(psi):
#     Psi[reshape_index(k, sa, sb)] = m

Psi = np.reshape(psi, (2, 2**(n-1)), order='F')

# print(Psi)

print('- '*10)

As = []

# Step 0
# SVD
u, langdas, vh = np.linalg.svd(Psi, full_matrices=False)
# print(u, '\n', langdas, '\n', vh)
# print(u.shape, langdas.shape, vh.shape)

Ai = u
psi = langdas[:, np.newaxis]*vh
# print(u.shape, psi.shape)
As.append(Ai)

# Steps 1..n-2
for j in range(1, n-1):

    shape = psi.shape

    Psi = np.empty((shape[0]*2, shape[1]//2))
    # print(Psi.shape)
    for k, row in enumerate(psi):
        # print(Psi[2*k:2*k+2].shape)
        Psi[2*k:2*k+2] = np.reshape(row, (2,  shape[1]//2), order='F')

    u, langdas, vh = np.linalg.svd(Psi, full_matrices=False)

    Ai = np.array([u[0::2], u[1::2]])
    psi = langdas[:, np.newaxis]*vh
    # print(u.shape, psi.shape)

    As.append(Ai)
    # print(j)

# print('+ '*10)

# Step n-1
# print(psi.shape)
Ai = np.transpose(psi)
# Ai = Psi
As.append(Ai)

# print(As)

# print('+ '*10)
# for Ai in As:
#     print(Ai[0].shape)

# Calculate entropy
normalization = np.sum(langdas**2)
S = -np.sum(langdas**2*np.log(langdas**2))

# print('+ '*10)
Psi_test = []
for j in range(2**n):
    string = bin(j)[2:]
    string = '0'*(n-len(string))+string
    bin_array = np.array([int(x) for x in string])[::-1]

    el = As[0][bin_array[0]]
    # print(el.shape)
    for k, m in enumerate(bin_array[1:-1]):
        # print(As[k+1][m].shape)
        el = np.matmul(el, As[k+1][m])
    # print(el.shape, As[-1][bin_array[-1]].shape)

    el = np.dot(el, As[-1][bin_array[-1]])

    Psi_test.append(el)

Psi_test = np.array(Psi_test)
print(Psi_test, sum(Psi_test**2))




# plt.plot(ns, Ss, '.-')
#
# plt.show()

