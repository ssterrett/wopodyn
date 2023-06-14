"""Figures for manuscript

"""

#%%
#Imports
import os
from pathlib import Path

import matplotlib
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from matplotlib.gridspec import GridSpec

# style specs
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
from wopodyn import utils

import pickle

#%%
# 
raw_dir = Path('./data/raw/swimming')
suffix = 'N2_L1'
path = raw_dir / suffix
angles=[]
files=[]
for count, file in enumerate(path.glob('*.txt')):
    try:
        # Data Pre-processing/cleaning step to remove header of multiple lines (at times)
        with open(file) as curr_file:
            lines = curr_file.readlines()
            lines = [line.rstrip() for line in lines]
        ind = 0
        for i in range(len(lines)):
            if lines[i][:4] == 'Time':
                ind = i
                break
        # Actually loading the data from .txt
        angle = np.loadtxt(file, dtype='float', skiprows=ind+1)
        # A check to make sure that the data is 10 dimensional
        if len(angle[0]) < 11:
            angles.append(angle)
            print(f'{file} missing time')
        else:
            angles.append(angle[:, 1:])  # slice out time column
        files.append(file)
    except UnicodeDecodeError:
            print(f'{file} has unknown character')
    except ValueError as e:
            print(f'{file} has missing rows')
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
# file = open('pca.obj', 'rb')
# pca = pickle.load(file)
# pcs = pca.fit_transform(stds) #fit and transform the angles data

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
# %%
fig, ax = plt.subplots(figsize=(9,9))
rotation_in_degrees = 45
tr = transforms.Affine2D().rotate_deg(rotation_in_degrees)

im1 = ax.imshow(Z, origin='lower', cmap='viridis',
          extent=[gmin, gmax, gmin, gmax],
          transform=tr + ax.transData)
ax.set(xlim = [-4, 4], 
        xticks = np.arange(-3, 4),
        ylim = [-4, 4], 
        yticks = np.arange(-3, 4),
        xlabel = 'Eigenworm 1', 
        ylabel = 'Eigenworm 2')
fig.colorbar(im1, ax=ax)
# figname = Path('reports/figures/swimhisto.pdf')
# fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")
# %%
# file = open('pca.obj', 'rb')
# pca = pickle.load(file)
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
# fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")
# %%
filename = Path('data/raw/10secvids/032522_N2_L1_swim_0008_10sec.txt')
data = np.loadtxt(filename, dtype= 'float', skiprows=1)
time = data[:,0]
angles = data[:,1:]
fig, ax = plt.subplots(figsize=(4,1.3))

heatmap(data[:,0], angles, fig=fig, ax=ax)
figname = Path('reports/figures/10seckymo.pdf')
# fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")

# %%
scaler = StandardScaler() #initialize a standarizing object
std = scaler.fit_transform(angles) #normalize the data

pcs = pca.transform(std)

# %%
fig = plt.figure(figsize=(4, 1.2))
gs = GridSpec(1,3, figure=fig)
ax1 = fig.add_subplot(gs[0,:2])
ax2 = fig.add_subplot(gs[0, 2])

ax = ax1
heatmap(data[:,0], angles*0.1, fig=fig, ax=ax)
line1 = ax.axvline(x=0, linewidth=1, c='k')

ax = ax2
pc1 = pcs[:,0]
pc2 = pcs[:,1]
im1 = ax.imshow(Z, origin='lower', cmap='viridis',
          extent=[gmin, gmax, gmin, gmax],
          transform=tr + ax.transData)
ax.set(xlim = [-4, 4], 
        xticks = np.arange(-3, 4),
        ylim = [-4, 4], 
        yticks = np.arange(-3, 4),
        xlabel = 'Eigenworm 1', 
        ylabel = 'Eigenworm 2')
scat1 = ax.scatter(pc1[0:5], pc2[0:5], c=np.arange(5), cmap='Greys')
fig.tight_layout()
#
def animate(i):
    line1.set_xdata(i)
    x_i = pc1[i:i+5]
    y_i = pc2[i:i+5]
    scat1.set_offsets(np.c_[x_i, y_i])

# 

anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(pc1))
anim.save(Path('./reports/figures/N2_L1_Swim_10_sync.mp4'),
          writer='ffmpeg',
          fps=int(len(pc1)/(time[-1]-time[0])),
          dpi=300)

# %%
# fig, ax = plt.subplots(figsize=(4, 1.3))
# ax.plot(pcs[:,:4])
# ax.set(xlabel='Time (s)',
#         xticks=[0, 140],
#         xticklabels=['0', '10'],
#         xlim=[0, pcs.shape[0]],
#         ylabel='Eigenworm Loadings',
#         )

# figname = Path('reports/figures/10secload.pdf')
# fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")

# %%
# TODO make auto repeat over swim and crawl
raw_dirs = ['data/raw/swimming', 'data/raw/crawling']
save_dirs = ['reports/figures/swimdevpcs.pdf','reports/figures/crawldevpcs.pdf']
strain = 'N2'
ages = ['_L1', '_LateL1', '_L2', '_L3', '_L4', '_Adult']

for locoind in range(2):
        # angles is a list where each element is the data from one age group
        # filenames is a list where each element is an array of filenames from one age group
        angles, filenames = [], []

        # Import the data
        for age in ages:
                path = Path(raw_dirs[locoind]) / (strain + age)
                # Initialize the angle and filename element for this particular age group, which will be appended to the overall angles, filenames
                age_angle = []
                age_files = []
                # Loops through all .txt files in the subfolder made for this strain, age, and mode
                for count, file in enumerate(path.glob('*.txt')):
                        try:
                                # Data Pre-processing/cleaning step to remove header of multiple lines (at times)
                                with open(file) as curr_file:
                                        lines = curr_file.readlines()
                                        lines = [line.rstrip() for line in lines]
                                ind = 0
                                for i in range(len(lines)):
                                        if lines[i][:4] == 'Time':
                                                ind = i
                                                break
                                # Actually loading the data from .txt
                                angle = np.loadtxt(file, dtype='float', skiprows=ind+1)
                                # A check to make sure that the data is 10 dimensional
                                if len(angle[0]) < 11:
                                        age_angle.append(angle)
                                        print(f'{file} missing time')
                                else:
                                        age_angle.append(angle[:, 1:])  # slice out time column
                                age_files.append(file)
                        except UnicodeDecodeError:
                                print(f'{file} has unknown character')
                        except ValueError as e:
                                print(f'{file} has missing rows')
                # Append the age group specific angle/filenames to the overall list
                angles.append(age_angle)
                filenames.append(age_files)

        # standardize/scale the data
        all_eigenworms = []
        for ind, age_angles in enumerate(angles):
                stds = []
                for angle in age_angles:
                        scaler = StandardScaler()  # initialize a standardizing object
                        stds.append(scaler.fit_transform(angle))  # normalize the data
                stds = np.vstack(stds)  # stack to (n_frames, n_segments) for all data
                pca = PCA(n_components=10)  # init pca object
                pcs = pca.fit_transform(stds)  # fit and transform the angles data
                all_eigenworms.append(pca.components_)

        flips_all = [np.array([[1,-1,-1,-1,1,-1],
        [-1,1,-1,1,1,1],
        [-1,-1,1,-1,-1,1],
        [-1,-1,-1,1,-1,-1],
        [1,-1,-1,1,1,1],
        [1,1,-1,-1,1,1]]),
        np.array([[1, 1, 1, -1, 1, 1],
        [1, -1, 1, 1, -1, 1],
        [-1, 1, 1, 1, -1, -1],
        [-1, -1, -1, -1, 1, -1],
        [1,1,1,1,-1,-1],
        [-1,-1,-1,-1,1,-1]])]
        fig, axes = plt.subplots(2, 2, figsize=(3,2))
        axes = axes.ravel()
        for i, eigenworms in enumerate(all_eigenworms):
                for j, ax in enumerate(axes):
                        ax.plot(eigenworms[j]*flips_all[locoind][j,i], '--', label=ages[i][1:])
                        ax.set(xticks=range(0,9,2),
                                ylim=[-0.6, 0.6],
                                yticks=[-0.5, 0, 0.5])
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(save_dirs[locoind], dpi=300, transparent=True)

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
