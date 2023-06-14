#%%
#Imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.backends.backend_pdf
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
from pathlib import Path

# from src.vizualization import heatmap

import pickle

# %%
# TODO make auto repeat over swim and crawl
raw_dir = 'data/raw/crawling'
strain = 'N2'
ages = ['_L1', '_LateL1', '_L2', '_L3', '_L4', '_Adult']

# angles is a list where each element is the data from one age group
# filenames is a list where each element is an array of filenames from one age group
angles, filenames = [], []

# Import the data
for age in ages:
    path = Path(raw_dir) / (strain + age)
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
all_cumvars = []
for ind, age_angles in enumerate(angles):
    stds = []
    for angle in age_angles:
        scaler = StandardScaler()  # initialize a standardizing object
        stds.append(scaler.fit_transform(angle))  # normalize the data
    stds = np.vstack(stds)  # stack to (n_frames, n_segments) for all data
    pca = PCA(n_components=10)  # init pca object
    pcs = pca.fit_transform(stds)  # fit and transform the angles data
    all_eigenworms.append(pca.components_)
    all_cumvars.append(np.cumsum(np.hstack(([0], pca.explained_variance_ratio_))))

flips_swim = np.array([[1,-1,-1,-1,1,-1],
  [-1,1,-1,1,1,1],
  [-1,-1,1,-1,-1,1],
  [-1,-1,-1,1,-1,-1],
  [1,-1,-1,1,1,1],
  [1,1,-1,-1,1,1]])
flips_crawl = np.array([[1, 1, 1, -1, 1, 1],
  [1, -1, 1, 1, -1, 1],
  [-1, 1, 1, 1, -1, -1],
  [-1, -1, -1, -1, 1, -1],
  [1,1,1,1,-1,-1],
  [-1,-1,-1,-1,1,-1]])
fig, axes = plt.subplots(2, 3, figsize=(3,2))
axes = axes.ravel()
for i, eigenworms in enumerate(all_eigenworms):
    for j, ax in enumerate(axes):
        ax.plot(eigenworms[j]*flips_crawl[j,i], '--', label=ages[i][1:])
        ax.set(xticks=range(0,9,2))
axes[0].legend()
fig.tight_layout()
# fig.savefig('reports/figures/crawldevpcs.pdf', dpi=300, transparent=True)
# %%
#
fig, ax = plt.subplots(figsize=(2, 2))
for i, cumvar in enumerate(all_cumvars):
  ax.plot(cumvar, '--', label=ages[i][1:])
ax.set(xlabel='Eigenworm',
        xlim=[0, 10],
        xticks=range(0, 11),
        xticklabels=[str(s) for s in range(0,11)],
        ylim=[0, 1.1],
        ylabel='Cumulative Variance \n explained',
        title='Crawling')
ax.axhline(y=1, c='k', ls='--')
ax.legend()
fig.savefig('reports/figures/crawldevcumvar.pdf', dpi=300, transparent=True, bbox_inches="tight")

#%%
# OLD STUFF
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
fig, ax = plt.subplots(figsize=(8,8))
im1 = ax.imshow(np.rot90(Z), cmap='viridis',
          extent=[xmin, xmax, ymin, ymax])
ax.set(xlim = [-5, 5], ylim = [-5, 5], xlabel = r"$PC_1$", ylabel = r"$PC_2$",
        title=suffix[1:])
filename = os.path.join(r"C:\Users\Scott\Documents\python\wopodyn\reports\figures\devpcs",
            suffix[1:] + ".png")
fig.savefig(filename)

# %%
suffixs = [ r"\N2_L1", r"\N2_LateL1", r"\N2_L2",  r"\N2_L3",  r"\N2_L4", r"\N2_Adult"]
for suffix in suffixs:
    raw_dir = r"C:\Users\Scott\Documents\python\wopodyn\data\raw"
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

    fig, ax = plt.subplots(figsize=(8,8))
    im1 = ax.imshow(np.rot90(Z_L1), cmap='viridis',
              extent=[xmin, xmax, ymin, ymax])
    ax.set(xlim = [-5, 5], ylim = [-5, 5], xlabel = r"$PC_1$", ylabel = r"$PC_2$",
            title=suffix[1:])
    filename = os.path.join(r"C:\Users\Scott\Documents\python\wopodyn\reports\figures\devpcs",
                suffix[1:] + ".png")
    fig.savefig(filename)

# %%
# Plot all eigenworms for each stage
fig, axes = plt.subplots(4,1, figsize=(9,12))
axes = axes.ravel()
suffixs = [ r"\N2_L1", r"\N2_LateL1", r"\N2_L2",  r"\N2_L3",  r"\N2_L4", r"\N2_Adult"]
for i, suffix in enumerate(suffixs):
    raw_dir = r"C:\Users\Scott\Documents\python\wopodyn\data\raw"
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

    pca = PCA(n_components=10) #init pca object
    pcs = pca.fit_transform(stds) #fit and transform the angles data

    ax = axes[0]
    flips = [-1, 1, -1, 1, 1, -1]
    ax.plot(pca.components_[0,:]*flips[i], label=suffix[1:])
    ax.set(ylim=[-1, 1], xticks=[], title='Eigenworm 1')

    ax = axes[1]
    flips = [1, 1, -1, -1, 1, 1]

    ax.plot(pca.components_[1,:]*flips[i])
    ax.set(ylim=[-1, 1], xticks=[], title='Eigenworm 2')

    ax = axes[2]
    flips = [-1, -1, 1, 1, 1, -1]

    ax.plot(pca.components_[2,:]*flips[i])
    ax.set(ylim=[-1, 1], xticks=[], title='Eigenworm 3')

    ax = axes[3]
    flips = [1, -1, -1, 1, 1, 1]

    ax.plot(pca.components_[3,:]*flips[i])
    ax.set(ylim=[-1, 1], xticks=[], title='Eigenworm 4')
fig.legend(loc='lower right')
