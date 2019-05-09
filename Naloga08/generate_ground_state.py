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
import scipy.sparse as sparse
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

s0 = [[1, 0], [0, 1]]
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


fig = plt.figure()

ns = range(4, 22, 2)
periodic = True

# Construct Hamiltonian

ground_states = []

for n in ns:
    H = sparse.csc_matrix((2**n, 2**n))

    for k in range(n-1):
        H += sparse.kron(sparse.identity(2**k), sparse.kron(h2, sparse.identity(2**(n-k-2))))

    # Periodic boundary condition
    if periodic:
        for s in (sx, sy, sz):
            H += np.real(sparse.kron(s, sparse.kron(sparse.identity(2**(n-2)), s)))

    # Get ground state
    eig, vec = sparse.linalg.eigsh(H)
    psi = vec[:, 0]

    sparse_vec = sparse.csc_matrix(np.where(np.abs(psi) > 1e-10, psi, 0))
    print(n, 2**n, sparse_vec.count_nonzero())

    ground_states.append(psi)

np.save('ground_states_per', ground_states)


# print(np.log2(psi.size))
