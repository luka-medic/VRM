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

periodic = 'per'

for i in range(5):
    print(i)

ns = [4, 6, 8, 10, 12, 14, 16, 18, 20,]

fig = plt.figure(figsize=(8, 6))

clr = ['#1f77b4', '#ff7f0e', '#2ca02c']
for j, periodic in enumerate(['per', 'non', 'gauss']):

    color = iter(cm.gnuplot(np.linspace(0.95, 0.05, len(ns))))
    for nidx, n in enumerate(ns):
        c = next(color)
        data = np.load('data/data2_2_' + periodic + str(n) + '.npy')
        ns_part, Ss = data[0], data[1]

        if nidx == 0:
            plt.plot(ns_part, Ss, '.-', color=clr[j], label=periodic)
        else:
            plt.plot(ns_part, Ss, '.-', color=clr[j],)

        if j==2:
            if nidx == 0:
                plt.plot(ns_part, n*np.log(2)*(0.5-np.abs(0.5-ns_part)), '--k', label='max')
            else:
                plt.plot(ns_part, n*np.log(2)*(0.5-np.abs(0.5-ns_part)), '--k')


plt.title('Entropije prepletenosti $S$')
# plt.title('Entropija prepletenosti $S$ za normalno porazdeljene\nkoeficiente val. funkcije $\Psi$')

plt.xlabel('$n_A$ / $n$   (1 $-$ $n_B$ / $n$)')
plt.ylabel('S')
plt.legend()


print('sprememba2')



fig.savefig('slika3_4.pdf')

plt.show()

