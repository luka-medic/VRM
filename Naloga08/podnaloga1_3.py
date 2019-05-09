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


def reshape_index(num, sa, sb):
    string = bin(num)[2:]
    string = '0'*(n-len(string))+string
    bin_array = np.array([int(x) for x in string])[::-1]

    i = sum([c*2**k for k, c in enumerate(bin_array[sa])])
    j = sum([c*2**k for k, c in enumerate(bin_array[sb])])

    return (i, j)


fig = plt.figure()

for periodic in ('per', 'non'):
    ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    Ss = []

    ground_states = np.load('ground_states_'+periodic+'.npy')

    for ii, n in enumerate(ns):
        print(n)

        # Get ground state
        psi = ground_states[ii]

        # Reshape psi -> Psi
        #compact
        # sa, sb = np.array([k for k in range(n//2)]), np.array([k for k in range(n//2, n)])
        #noncompact
        sa, sb = np.array([2*k for k in range(n//2)]), np.array([2*k+1 for k in range(n//2)])
        sa, sb = np.array([4*k for k in range(n//4)]+[4*k+1 for k in range(n//4)]),\
                 np.array([4*k+2 for k in range(n//4)]+[4*k+3 for k in range(n//4)])

        a, b = 2**sa.size, 2**sb.size
        Psi = np.zeros((a, b))

        for k, m in enumerate(psi):
            Psi[reshape_index(k, sa, sb)] = m

        # SVD
        u, langdas, vh = np.linalg.svd(Psi)

        # Calculate entropy
        normalization = np.sum(langdas**2)
        S = -np.sum(langdas**2*np.log(langdas**2))

        Ss.append(S)

    np.save('data1_3_noncompact2_'+periodic, Ss)

    plt.plot(ns, Ss, '.-')



print('show')
plt.show()

