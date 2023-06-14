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

#calculate cumvar, eigenworms, and PC histogram
cumvar = np.cumsum(np.hstack(([0], pca.explained_variance_ratio_)))

kde = gaussian_kde(pcs[:,0:2].T)
xmin = np.min(pcs[:,0])
xmax = np.max(pcs[:,0])
ymin = np.min(pcs[:,1])
ymax = np.max(pcs[:,1])

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)
fig, ax = plt.subplots(figsize=(12,9))
im1 = ax.imshow(np.rot90(Z), cmap='viridis',
          extent=[xmin, xmax, ymin, ymax])
ax.set(xlim = [-5, 5], ylim = [-5, 5], xlabel = r"$PC_1$", ylabel = r"$PC_2$")
plt.colorbar(im1, ax=ax)
ax.scatter(x,y, c=np.arange(50))
# %%
rad = 2.1
x = np.sin(np.linspace(0, 2*np.pi, 50))*rad
y = np.cos(np.linspace(0, 2*np.pi, 50))*rad
positions = np.vstack([x,y])

dens = kde(positions)
dens_swim = dens
# dens_crawl = dens
# %%
fig, ax = plt.subplots(figsize=(12,9))
ax.plot(dens_swim, label='Swim')
ax.plot(dens_crawl, label='Crawl')
ax.set(xlabel='Angle', xticks=[0, 25, 50], xticklabels=['0', '$\pi$', '$2\pi$'],
        ylabel='Density', title='Ring Density Comparison')
ax.legend(title='Behavior')
