#%%
#Imports
import os
from pathlib import Path

import matplotlib
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# plt.style.use('seaborn-paper')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 1
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 6
matplotlib.rcParams['axes.labelpad'] = 0
matplotlib.rcParams['xtick.labelsize'] = 6
matplotlib.rcParams['xtick.major.size'] = 2
matplotlib.rcParams['xtick.major.width'] = 0.3
matplotlib.rcParams['ytick.labelsize'] = 6
matplotlib.rcParams['ytick.major.size'] = 2
matplotlib.rcParams['ytick.major.width'] = 0.3

import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from wopodyn.vizualization import heatmap

#%%
# 
#compile data from all files in directory
raw_dir = Path('./data/raw')
suffix = 'N2_Adult'
path = raw_dir / suffix
angles = []
filenames = []
for count, file in enumerate(path.glob('*')):
    filenames.append(file)
    angle = np.loadtxt(file, dtype= 'float', skiprows=1)
    angles.append(angle[:,1:]) #slice out time column
print(f'{count+1} files in directory for {suffix}')
# %%
#scale and perform pca on data
stds = []
for angle in angles:
  scaler = StandardScaler() #initialize a standarizing object
  stds.append(scaler.fit_transform(angle)) #normalize the data
stds = np.vstack(stds) #stack to (n_frames, n_segments) for all data

pca = PCA(n_components=10) #init pca object
pcs = pca.fit_transform(stds) #fit and transform the angles data
# %%
kde = gaussian_kde(pcs[:,0:2].T)
xmin = np.min(pcs[:,0])
xmax = np.max(pcs[:,0])
ymin = np.min(pcs[:,1])
ymax = np.max(pcs[:,1])

gmin = np.min([xmin, ymin]) # left bottom min
gmax = np.max([xmax, ymax]) # right top min

fmin = np.min([abs(gmin), abs(gmax)]) # min for lim

X, Y = np.mgrid[gmin:gmax:100j, gmin:gmax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

fig, ax = plt.subplots(figsize=(1.3,1.3))
im1 = ax.imshow(np.rot90(Z), cmap='viridis',
          extent=[gmin, gmax, gmin, gmax])
ax.set(xlim = [-4, 4], 
        xticks = np.arange(-3, 4),
        ylim = [-4, 4], 
        yticks = np.arange(-3, 4),
        xlabel = 'Eigenworm 1', 
        ylabel = 'Eigenworm 2')
fig.colorbar(im1, ax=ax)
figname = Path('reports/figures/swimhisto.pdf')
fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")
# %%
fig, ax = plt.subplots(figsize=(1.3, 1.3))
cumvar = np.cumsum(np.hstack(([0], pca.explained_variance_ratio_)))
ax.hlines(y=cumvar[1:5],
          xmin=np.zeros(4),
          xmax=range(1,5),
          colors= plt.rcParams['axes.prop_cycle'].by_key()['color'][0:4],
          linestyles='dashed',
          linewidths=0.5)
ax.plot(cumvar, 'ko-')
ax.set( xlim=[0,10],
        xticks = np.arange(11),
        xticklabels = [str(tick) for tick in np.arange(11)],
        xlabel = "Eigenworms",
        ylim=[0, 1.1],
        ylabel = "Cumulative Variance Explained")
figname = Path('reports/figures/swimcumvarv2.pdf')
fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")
# %%
filename = Path('data/10sec_082522_N2_Adult_swim_0005.txt')
data = np.loadtxt(filename, dtype= 'float', skiprows=1)
angles = data[:,1:]*0.1
fig, ax = plt.subplots(figsize=(4,1.3))

heatmap(data[:,0], angles, fig=fig, ax=ax)
figname = Path('reports/figures/10seckymo.pdf')
fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")

# %%
scaler = StandardScaler() #initialize a standarizing object
std = scaler.fit_transform(angles) #normalize the data

pcs = pca.transform(std)

fig, ax = plt.subplots(figsize=(4, 1.3))
ax.plot(pcs[:,:4])
ax.set(xlabel='Time (s)',
        xticks=[0, 140],
        xticklabels=['0', '10'],
        xlim=[0, pcs.shape[0]],
        ylabel='Eigenworm Loadings',
        )

figname = Path('reports/figures/10secload.pdf')
fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")

# %%
raw_dir = Path('./data/raw')
suffix = 'N2_Adultcrawl'
path = raw_dir / suffix
angles = []
filenames = []
for count, file in enumerate(path.glob('*')):
    filenames.append(file)
    angle = np.loadtxt(file, dtype= 'float', skiprows=1)
    angles.append(angle[:,1:]) #slice out time column
print(f'{count+1} files in directory for {suffix}')

#scale and perform pca on data
stds = []
for angle in angles:
  scaler = StandardScaler() #initialize a standarizing object
  stds.append(scaler.fit_transform(angle)) #normalize the data
stds = np.vstack(stds) #stack to (n_frames, n_segments) for all data

pca = PCA(n_components=10) #init pca object
pcs = pca.fit_transform(stds) #fit and transform the angles data

# calculate density estimate
kde = gaussian_kde(pcs[:,0:2].T)
xmin = np.min(pcs[:,0])
xmax = np.max(pcs[:,0])
ymin = np.min(pcs[:,1])
ymax = np.max(pcs[:,1])

gmin = np.min([xmin, ymin]) # left bottom min
gmax = np.max([xmax, ymax]) # right top min

fmin = np.min([abs(gmin), abs(gmax)]) # min for lim

X, Y = np.mgrid[gmin:gmax:100j, gmin:gmax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

fig, ax = plt.subplots(figsize=(1.3,1.3))
im1 = ax.imshow(np.rot90(Z), cmap='viridis',
          extent=[gmin, gmax, gmin, gmax])
ax.set(xlim = [-fmin, fmin], 
        xticks = np.arange(-3, 4),
        ylim = [-fmin, fmin], 
        yticks = np.arange(-3, 4),
        xlabel = 'Eigenworm 1', 
        ylabel = 'Eigenworm 2')
fig.colorbar(im1, ax=ax)
figname = Path('reports/figures/crawlhisto.pdf')
fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")
# %%
