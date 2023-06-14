#%%
#Imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

kde = gaussian_kde(pcs[:,0:2].T)
xmin = np.min(pcs[:,0])
xmax = np.max(pcs[:,0])
ymin = np.min(pcs[:,1])
ymax = np.max(pcs[:,1])

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

# %%
raw_dir = r"C:\Users\Scott\Documents\python\wopodyn\data"
suffix = r"\N2_L1"
path = raw_dir + suffix
angles = []
filenames = []
for root, dirs, files in os.walk(path):
  for count, file in enumerate(files):
    filenames.append(os.path.join(root,file))
    angle = np.loadtxt(filenames[count], dtype= 'float', skiprows=1)
    angles.append(angle[:,1:]) #slice out time column
print(count, ' files in directory for ', suffix[1:])

stds = []
for angle in angles:
  scaler = StandardScaler() #initialize a standarizing object
  stds.append(scaler.fit_transform(angle)) #normalize the data
stds = np.vstack(stds) #stack to (n_frames, n_segments) for all data

pcs = pca.transform(stds) #fit and transform the angles data

kde = gaussian_kde(pcs[:,0:2].T)
xmin = np.min(pcs[:,0])
xmax = np.max(pcs[:,0])
ymin = np.min(pcs[:,1])
ymax = np.max(pcs[:,1])

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z_L1 = np.reshape(kde(positions).T, X.shape)

# %%
#transform session
filename = r"C:\Users\Scott\Documents\python\wopodyn\data\N2_Adult.txt"
data = np.loadtxt(filename, dtype= 'float', skiprows=1)
start = 0
stop = 10
start_idx = (np.abs(data[:,0] - start)).argmin()
stop_idx = (np.abs(data[:,0] - stop)).argmin()
time = data[start_idx:stop_idx,0]; angle = data[start_idx:stop_idx,1:]
angle.shape
scaler = StandardScaler() #initialize a standarizing object
std = scaler.fit_transform(angle)
pcs = pca.transform(std)

# %%
#plot heatmap, loadings, pc1vpc2

fig = plt.figure(figsize=(12,9), constrained_layout=True)
gs = GridSpec(2,3, figure=fig)
ax1 = fig.add_subplot(gs[0,:2])
ax2 = fig.add_subplot(gs[1,:2])
ax3 = fig.add_subplot(gs[0,2])

ax = ax1
heatmap(time, angle, fig=fig, ax=ax, timestep = 2)
line1 = ax.axvline(x=0, linewidth=4, c='k')

ax = ax2
loadings = ax.plot(time, pcs[:,:4], linewidth=2, alpha=0.7)
line2 = ax.axvline(x=0, linewidth=4, c='k')
ax.set(xlim=[min(time), max(time)], xlabel='Time (s)', ylabel='Loadings (au)',
        xticks=[0, 2, 4, 6, 8])
ax.legend(loadings, ('PC1', 'PC2', 'PC3', 'PC4'))
# fig.suptitle(os.path.split(filename)[1])

ax = ax3
pc1 = pcs[:,0]
pc2 = pcs[:,1]
im1 = ax.imshow(np.rot90(Z_L1), cmap='viridis',
          extent=[xmin, xmax, ymin, ymax])
ax.set(xlim = [-5, 5], ylim = [-5, 5], xlabel = r"$PC_1$", ylabel = r"$PC_2$")
scat1 = ax.scatter(pc1[0:5], pc2[0:5], c=np.arange(5), cmap='Greys')
# plt.tight_layout()
#

# %%


def animate(i):
    line1.set_xdata(i)
    line2.set_xdata(time[i])
    x_i = pc1[i:i+5]
    y_i = pc2[i:i+5]
    scat1.set_offsets(np.c_[x_i, y_i])

#
anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(pc1))
anim.save(r"C:\Users\Scott\Documents\python\wopodyn\reports\figures\N2_L1_10.mp4",
            fps=int(len(pc1)/10), dpi=300)

# %%
fig, ax = plt.subplots(figsize=(12,9))
loadings = ax.plot(time, pcs[:,:4], linewidth=5, alpha=0.7)
ax.set(xlim=[2, 4], xlabel='Time (s)', ylabel='Loadings (au)')
