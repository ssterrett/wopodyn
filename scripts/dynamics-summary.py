# %% codecell
#Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns
sns.set_context('poster')
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal
from sklearn.manifold import TSNE

import src.vizualization as viz

#%%
# load data, scale and pc, wavelet expansion
for i in range(1,len(dfs)):

    df = dfs[ids[i]].to_numpy()
    if np.sum(np.isnan(df)) == 0:
        time = df[:, 0]
        dat = df[:, 1:]
        dat = StandardScaler().fit_transform(dat)
        dat_pc = pca.transform(dat)
        dat_pc = dat_pc.T
        # data is at 100 Hz, get 15 scales spaced from 100 ms to 1 s
        widths = np.linspace(2, 20, 10)
        wavelets_init = signal.cwt(dat_pc[0,:], signal.ricker, widths)

        for x in range(9):
            cwtmatr = signal.cwt(dat_pc[x+1,:], signal.ricker, widths)
            wavelets_init = np.concatenate([wavelets_init, cwtmatr])
            # concatenate and z-score wavelets
            wavelets_stand = StandardScaler().fit_transform(wavelets_init)
            wavelets_all_norm = np.vstack((wavelets_all_norm, wavelets_stand.T))

# tsne dim reduction
tsne_dyn = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(wavelets_all_norm)

#%%
# animated plot of pose dynamics
# to plot an individual session, need start and stop inds for that session

fig, ax = plt.subplots(figsize=(5, 3))

x = tsne_dyn[inds[0]:inds[1], 0]
y = tsne_dyn[inds[0]:inds[1], 1]

ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)), xticks=[], yticks=[], aspect='equal')
ax.hist2d(tsne_dyn[:,0], tsne_dyn[:,1], bins=50)
scat = ax.scatter(x[0:10:2], y[0:10:2], c=range(5), cmap='binary', s=30)

def animate(i):
    x_i = x[i:i+5]
    y_i = y[i:i+5]
    ax.set_title(str(int(i/len(x)*60))+'s')
    scat.set_offsets(np.c_[x_i, y_i])
# %% codecell
anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(x))
