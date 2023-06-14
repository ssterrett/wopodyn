#%%
#Imports
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.backends.backend_pdf
plt.style.use('seaborn-talk')
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde, binned_statistic_2d
from scipy.signal import hilbert


from wopodyn.vizualization import heatmap

import pickle
import ssm
#%%
raw_dir = Path('./data/raw/swimming')
suffix = 'N2_Adult'
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

#scale and perform pca on data
stds = []
for angle in angles:
    scaler = StandardScaler() #initialize a standarizing object
    std = scaler.fit_transform(angle) #fit and transform the data
    stds.append(std) #normalize the data

stds_all = np.vstack(stds) #stack to (n_frames, n_segments) for all data
pca = PCA(n_components=10) #init pca object
pcs = pca.fit_transform(stds_all) #fit and transform the angles data 

pcs = []
for std in stds:
     pcs.append(pca.transform(std))

data = []
for pc in pcs:
     data.append(pc[:,:4])
# %%
# fit arhmm
arhmm = ssm.HMM(K=5, D=4, observations="ar")

_, _ = arhmm.fit(data)
# %%
for i, file in enumerate(files):
     if i < 10:
        fig, axs = plt.subplots(2, 1, figsize=(12,9))
        ax = axs[0]
        ax.plot(data[i][:,0])
        ax.plot(data[i][:,1])
        ax.set(xlim=[0,data[i].shape[0]])

        zs = arhmm.most_likely_states(data[i])
        ax = axs[1]
        ax.imshow(zs[None,:], aspect='auto')

        ax.set(title=file)

# %%
fig, ax = plt.subplots(figsize=(12,9))

for i, file in enumerate(files):
    zs = arhmm.most_likely_states(data[i])
    ax.scatter(data[i][:,0], data[i][:,1], c=zs, alpha=0.2)
# %%
# warped arhmm??