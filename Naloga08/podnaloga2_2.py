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

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

periodic = 'per'

n, nidx = 8, 2
ns = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
for nidx, n in enumerate(ns):
    print(n)

    ns_part = [i/n for i in range(n+1)]

    Ss = [0]
    As = []

    # Get ground state
    # ground_states = np.load('ground_states_'+periodic+'.npy')
    # psi = ground_states[nidx]

    # Get random state
    psi = sample_spherical(1, 2**n)

    # Reshape psi -> Psi
    sa, sb = np.array([k for k in range(1)]), np.array([k for k in range(1, n)])
    a, b = 2**sa.size, 2**sb.size
    Psi = np.zeros((a, b))

    Psi = np.reshape(psi, (2, 2**(n-1)), order='F')

    # Step 0
    # SVD
    u, langdas, vh = np.linalg.svd(Psi, full_matrices=False)

    S = -np.sum(langdas**2*np.log(langdas**2))
    Ss.append(S)

    Ai = u
    psi = langdas[:, np.newaxis]*vh
    As.append(Ai)

    print(u.shape, psi.shape)

    # Steps 1..n-2
    for j in range(1, n-1):
        print(n, j)
        shape = psi.shape

        Psi = np.empty((shape[0]*2, shape[1]//2))
        for k, row in enumerate(psi):
            Psi[2*k:2*k+2] = np.reshape(row, (2,  shape[1]//2), order='F')

        u, langdas, vh = np.linalg.svd(Psi, full_matrices=False)

        S = -np.sum(langdas ** 2 * np.log(langdas ** 2))
        Ss.append(S)

        Ai = np.array([u[0::2], u[1::2]])
        psi = langdas[:, np.newaxis]*vh
        As.append(Ai)

        # print(u.shape, psi.shape)


    # Step n-1
    Ai = np.transpose(psi)
    As.append(Ai)

    # Calculate entropy
    normalization = np.sum(langdas**2)
    S = -np.sum(langdas**2*np.log(langdas**2))


    # Test
    # Psi_test = []
    # for j in range(2**n):
    #     string = bin(j)[2:]
    #     string = '0'*(n-len(string))+string
    #     bin_array = np.array([int(x) for x in string])[::-1]
    #
    #     el = As[0][bin_array[0]]
    #     # print(el.shape)
    #     for k, m in enumerate(bin_array[1:-1]):
    #         # print(As[k+1][m].shape)
    #         el = np.matmul(el, As[k+1][m])
    #     # print(el.shape, As[-1][bin_array[-1]].shape)
    #
    #     el = np.dot(el, As[-1][bin_array[-1]])
    #
    #     Psi_test.append(el)
    # Psi_test = np.array(Psi_test)


    Ss.append(0)

    plt.plot(ns_part, Ss, '.-')

    np.save('data/data2_2_'+'gauss'+str(n), np.array([ns_part, Ss]))

plt.show()

