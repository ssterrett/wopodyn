#%%
#Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.backends.backend_pdf
plt.style.use('seaborn-talk')
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde

from src.vizualization import heatmap

import pickle

#%%
#compile data from all files in directory
raw_dir = r"C:\Users\Scott\Documents\python\wopodyn\data\raw"
suffix = r"\N2_Adult"
path = raw_dir + suffix
angles = []
filenames = []
for root, dirs, files in os.walk(path):
  for count, file in enumerate(files):
    filenames.append(os.path.join(root,file))
    angle = np.loadtxt(filenames[count], dtype= 'float', skiprows=1)
    angles.append(angle[:,1:]) #slice out time column
print(count, ' files in directory for ', suffix[1:])

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
# fit individual session
# filename = r".\data\raw\N2_L1\011222_N2_L1_Swim_0012_W1.txt"
filename = r".\data\N2_Adult.txt"
data = np.loadtxt(filename, dtype= 'float', skiprows=1)
time = data[:,0]
angle = data[:,1:]
start = 0
stop = 10
start_idx = (np.abs(time - start)).argmin()
stop_idx = (np.abs(time - stop)).argmin()

scaler = StandardScaler() #initialize a standarizing object

std = scaler.fit_transform(angle)
pcs = pca.transform(std)

# %%
fig, ax = plt.subplots(figsize=(12,9))
# start_idx=0; stop_idx=len(time)
ax.plot(time[start_idx:stop_idx], pcs[start_idx:stop_idx,0], label='PC1')
ax.plot(time[start_idx:stop_idx], pcs[start_idx:stop_idx,1], label='PC2')
ax.plot(time[start_idx:stop_idx], pcs[start_idx:stop_idx,2], label='PC3')
ax.plot(time[start_idx:stop_idx], pcs[start_idx:stop_idx,3], label='PC4')
ax.set(xlabel='Time (s)', ylabel='Loadings (au)', xlim=[time[start_idx], time[stop_idx-1]])
ax.legend(loc='lower left')

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

fig.savefig(r".\reports\N2Adult10.svg")
