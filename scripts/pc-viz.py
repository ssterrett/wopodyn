#%%
#Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use('seaborn-talk')
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde

from src.vizualization import heatmap


#%%
#compile data from all files in directory
# path = "/home/user/Documents/python/worm-pose/data/raw/N2_Adult"
path = r"C:\Users\Scott\Documents\python\wopodyn\data\raw\N2_L1"
angles = []
filenames = []
for root, dirs, files in os.walk(path):
  for count, file in enumerate(files):
    filenames.append(os.path.join(root,file))
    angle = np.loadtxt(filenames[count], dtype= 'float', skiprows=1)
    angles.append(angle[:,1:]) #slice out time column

#
#scale and perform pca on data
stds = []
for angle in angles:
  scaler = StandardScaler() #initialize a standarizing object
  stds.append(scaler.fit_transform(angle)) #normalize the data
stds = np.vstack(stds) #stack to (n_frames, n_segments) for all data

pca = PCA(n_components=10) #init pca object
pcs = pca.fit_transform(stds) #fit and transform the angles data


import matplotlib.backends.backend_pdf


#%%
pdf = matplotlib.backends.backend_pdf.PdfPages("N2_L1-pcs.pdf")
for file in filenames:
    data = np.loadtxt(file, dtype='float', skiprows=1)
    time = data[:,0]
    angle = data[:,1:]
    scaler = StandardScaler()
    angle_norm = scaler.fit_transform(angle)
    pcs = pca.transform(angle_norm)

    fig, axs = plt.subplots(2,1, figsize=(12,9), constrained_layout=True)
    axs = axs.ravel()
    ax = axs[0]
    heatmap(time, angle, fig=fig, ax=ax)

    ax = axs[1]
    lines =ax.plot(time, pcs[:,:4])
    ax.set(xlim=[min(time), max(time)])
    ax.legend(lines, ('PC1', 'PC2', 'PC3', 'PC4'))
    fig.suptitle(os.path.split(file)[1])
    pdf.savefig(fig)

pdf.close()
#%%

os.path.split(file)[1]
C = np.cov(stds.T)
plt.imshow(C)
