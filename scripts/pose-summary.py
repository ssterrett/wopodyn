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


#%%
#compile data from all files in directory
path = "/home/user/Documents/python/worm-pose/data/raw/egl-20_Adult"
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

#
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

#
#plotting
fig = plt.figure(figsize=(12,9))
gs = GridSpec(2,2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])

#cumulative variance explained
ax = ax1
ax.plot(cumvar)
ax.set(xticks = np.arange(11, step=2),
    xticklabels = ['0', '2', '4', '6', '8', '10'], xlabel = "Num Components",
    ylabel = "Cumulative Variance Explained")

#2D PC distribution
ax=ax2
im1 = ax.imshow(np.rot90(Z), cmap='viridis',
          extent=[xmin, xmax, ymin, ymax])
ax.set(xlim = [-5, 5], ylim = [-5, 5], xlabel = r"$PC_1$", ylabel = r"$PC_2$")
plt.colorbar(im, ax=ax)

#eigenworms
ax = ax3
im2 = ax.imshow(pca.components_.T, cmap='bwr')
ax.set(xlabel="Eigenworm", ylabel = "Segment")

fig.suptitle("egl-20 Adult Posture Summary", size=20)

# plt.savefig("egl-20adult.png", dpi=300)
