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


fig = plt.figure()

ns = [4, 6, 8, 10, 12, 14, 16, 18, 20]

for periodic in ('per', 'non'):
    Ss = np.load('data1_3_'+periodic+'.npy')

    plt.plot(ns, Ss, '.-', label=periodic)

plt.title('Entropija prepletenosti $S$ za kompaktno\nbiparticijo $n_A = n_B$ v odvisnosti od dolžine verige $n$')
# plt.title('Entropija prepletenosti $S$ za nekompaktno\nbiparticijo $n_A = n_B$ v odvisnosti od dolžine verige $n$')
# plt.title('Entropija prepletenosti $S$ za nekompaktno\nbiparticijo parov $n_A = n_B$ v odvisnosti od dolžine verige $n$')

plt.xlabel('n')
plt.ylabel('S')
plt.legend()

plt.tight_layout(pad=1.06)

fig.savefig('slika2_1.pdf')

plt.show()

