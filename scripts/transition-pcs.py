#%%
#Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

import matplotlib.backends.backend_pdf
# plt.style.use('seaborn-talk')
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde

# %%
#%%
#compile data from all files in directory
raw_dir = r"C:\Users\Scott\Documents\python\wopodyn\data\raw\old"
suffix = r"\N2_Adult_transition"
path = raw_dir + suffix
angles = []
filenames = []
for root, dirs, files in os.walk(path):
  for count, file in enumerate(files):
    filenames.append(os.path.join(root,file))
    angle = np.loadtxt(filenames[count], dtype= 'float', skiprows=1)
    angles.append(angle[:,1:]) #slice out time column
print(count+1, ' files in directory for ', suffix[1:])

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

# segment covariance matrix
C = np.cov(stds.T) # data must be (n_segments, n_frames)

# %%
#summary plotting
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
# posture dynamics 3d rotation plot
file = filenames[2]
data = np.loadtxt(file, dtype='float', skiprows=1)
time = data[:,0]
angle = data[:,1:]
scaler = StandardScaler()
angle_norm = scaler.fit_transform(angle)
pcs = pca.transform(angle_norm)

x = pcs[:,0]
y = pcs[:,1]
z = pcs[:,2]

plt.plot(x[3000:])
plt.plot(y[3000:])
plt.plot(z[3000:])

# %%
fig = plt.figure(layout='tight', figsize=(2.25,2.25))
ax = fig.add_subplot(projection='3d')
ax.scatter(x[3000:3700:2], y[3000:3700:2], z[3000:3700:2], c='navy', s=1, alpha=0.5)
ax.scatter(x[3000:3700:2], y[3000:3700:2], np.ones_like(z[3000:3700:2])*min(z), c='navy', s=1, alpha=0.1)
ax.scatter(np.ones_like(x[3000:3700:2])*min(x), y[3000:3700:2], z[3000:3700:2], c='navy', s=1, alpha=0.1)

ax.scatter(x[4100:], y[4100:], z[4100:], c='darkorange', s=1, alpha=0.5)
ax.scatter(x[4100:], y[4100:], np.ones_like(z[4100:])*min(z), c='darkorange', s=1, alpha=0.1)
ax.scatter(np.ones_like(x[4100:])*min(x), y[4100:], z[4100:], c='darkorange', s=1, alpha=0.1)

ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)), zlim=(min(z), max(z)),
        xticks=[], yticks=[], zticks=[],
        xlabel='Eigenworm 1', ylabel='Eigenworm 2', zlabel='Eigenworm 3')
ax.dist = 11
# %%
fig.savefig('3D_transition.pdf', dpi=300, bbox_inches='tight')
# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)), zlim=(min(z), max(z)),
        xticks=[], yticks=[], zticks=[], xlabel="$PC_1$", ylabel="$PC_2$",
        zlabel="$PC_3$")

scat = ax.scatter(x[0:10:2], y[0:10:2], z[0:10:2], c=range(5), cmap='binary', s=30)

def animate(i):
    i += 1800
    if i < 2200:
        if i > 2100:
            ax.view_init(elev=(ax.elev - 0.2), azim=(ax.azim + 0.6))
    x_i = x[i:i+5]
    y_i = y[i:i+5]
    z_i = z[i:i+5]
    ax.set_title(str(int(i/len(x)*max(time)))+'s')
    if i < 2031:
        ax.plot3D(x_i[:-2], y_i[:-2], z_i[:-2], alpha=0.05, c='blue')
    elif (i > 2031) & (i<2298):
        ax.plot3D(x_i[:-2], y_i[:-2], z_i[:-2], alpha=0.05, c='teal')
    elif i > 2298:
        ax.plot3D(x_i[:-2], y_i[:-2], z_i[:-2], alpha=0.05, c='orange')


    scat._offsets3d = (x_i, y_i, z_i)

anim = animation.FuncAnimation(fig, animate, interval=30, frames=1199)

anim.save(r"C:\Users\Scott\Documents\python\wopodyn\reports\figures\transitions_rotate2.mp4",
            dpi=300)

# %%

