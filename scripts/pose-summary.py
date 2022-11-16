#%%
#Imports
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.backends.backend_pdf
plt.style.use('seaborn-talk')
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde

from src.vizualization import heatmap


#%%
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

# segment covariance matrix
C = np.cov(stds.T) # data must be (n_segments, n_frames)

#
reports_dir = Path('./reports/figures')
figname = reports_dir / (suffix + 'pcs.pdf')
# pdf = matplotlib.backends.backend_pdf.PdfPages(figname)

#plotting
fig = plt.figure(figsize=(12,9))
gs = GridSpec(2,2, figure=fig)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])

#cumulative variance explained
ax = ax1
ax.plot(cumvar)
ax.set(xticks = np.arange(11, step=2),
    xticklabels = ['0', '2', '4', '6', '8', '10'], xlabel = "Num Components",
    ylabel = "Cumulative Variance Explained")

# Covariance structure
ax = ax2
im2 = ax.imshow(C)
ax.set(xticks=[0,9], xticklabels=['1', '10'], xlabel='Segment',
        yticks=[0,9], yticklabels=['1', '10'], ylabel='Segment')
#2D PC distribution
ax=ax3
im1 = ax.imshow(np.rot90(Z), cmap='viridis',
          extent=[xmin, xmax, ymin, ymax])
ax.set(xlim = [-5, 5], ylim = [-5, 5], xlabel = r"$PC_1$", ylabel = r"$PC_2$")
plt.colorbar(im1, ax=ax)

#eigenworms
ax = ax4
im2 = ax.imshow(pca.components_.T, cmap='bwr')
ax.set(xlabel="Eigenworm", ylabel = "Segment")

fig.suptitle(suffix[1:] + " Posture Summary", size=20)

# %%
fig.savefig(Path('reports/figures/swim.png'), dpi=300)
# %%
# pdf.savefig(fig)
# plt.savefig("egl-20adult.png", dpi=300)

# plot heatmap and pcs for each trial
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
    fig.suptitle(file.parts[-1])
    pdf.savefig(fig)

pdf.close()
