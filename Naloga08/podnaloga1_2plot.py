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
from scipy.special import comb
import scipy.sparse as sparse

plt.rcParams['animation.ffmpeg_path'] = 'C:/FFmpeg/bin/ffmpeg'
plt.rc('text', usetex=0)
plt.rc('font', size=16, family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('legend', fontsize='small')
plt.rc('figure', figsize=(8, 6))
plt.rc('lines', linewidth=1.0)


ns = range(4, 22, 2)
es = [1, 2, 3, 4, 6, 8, 10, 12, 15, 16]

fig1 = plt.figure(figsize=(8, 6))

color = iter(cm.gnuplot(np.linspace(0.95, 0.05, len(es))))

for e in es:
    c = next(color)
    ratios1 = []
    ratios2 = []

    ground_states = np.load('ground_states_per.npy')

    for i, n in enumerate(ns):
        psi = ground_states[i]
        sparse_vec = sparse.csc_matrix(np.where(np.abs(psi) > 0.1**e, psi, 0))
        ratios1.append(comb(n, n//2)/2**n)
        ratios2.append(sparse_vec.count_nonzero()/2**n)

    ratios1 = np.array(ratios1)
    ratios2 = np.array(ratios2)

    # plt.plot(ns, ratios1, '.-')
    # plt.plot(ns, ratios2, '.-')
    plt.plot(ns, ratios2/ratios1, '.-', color=c, label='< 1e-'+str(e))
    # plt.plot(ns, ratios1-ratios2, '.-')

# plt.yscale('log')

plt.title(r'Razmerje števila amplitud osnovnega stanja $\psi_0$ večjih od $\epsilon$'
          '\n'
          r'in števila konfiguracij iz sektorja $n_{\uparrow}=n_{\downarrow}=n/2$')
plt.xlabel('n')
plt.ylabel('razmerje')
plt.legend(ncol=2)

plt.grid()

plt.tight_layout(pad=1.06)


fig2 = plt.figure(figsize=(8, 6))

color = iter(cm.gnuplot(np.linspace(0.95, 0.05, len(es))))

for e in es:
    c = next(color)

    ratios1 = []
    ratios2 = []

    ground_states = np.load('ground_states_per.npy')

    for i, n in enumerate(ns):
        psi = ground_states[i]
        sparse_vec = sparse.csc_matrix(np.where(np.abs(psi) > 0.1**e, psi, 0))
        ratios1.append(comb(n, n//2)/2**n)
        ratios2.append(sparse_vec.count_nonzero()/2**n)

    ratios1 = np.array(ratios1)
    ratios2 = np.array(ratios2)


    plt.plot(ns, ratios2, '.-', color=c, label='< 1e-'+str(e))
    # plt.plot(ns, ratios2/ratios1, '.-', label='< 1e-'+str(e))
    # plt.plot(ns, ratios1-ratios2, '.-')

plt.plot(ns, ratios1, 'x--k', label=r'sektor $n_{\uparrow}=n_{\downarrow}=n/2$', ms=10,)

plt.title(r'Razmerje števila amplitud osnovnega stanja $\psi_0$ večjih od $\epsilon$'
          '\n'
          r'in števila vseh konfiguracij $2^n$ (periodični r.p.)')
plt.xlabel('n')
plt.ylabel('razmerje')
plt.legend(loc='upper right', ncol=2)

plt.grid()

plt.tight_layout(pad=1.06)

fig1.savefig('slika1_1.pdf')
fig2.savefig('slika1_2.pdf')

plt.show()
