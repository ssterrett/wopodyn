#%%
#Imports
import os
from pathlib import Path

import matplotlib
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

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

#%%
#
raw_dir = 'data/raw'
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
stds = []
for age_angles in angles:
    for angle in age_angles:
        scaler = StandardScaler()  # initialize a standardizing object
        stds.append(scaler.fit_transform(angle))  # normalize the data
stds = np.vstack(stds)  # stack to (n_frames, n_segments) for all data

# Run PCA
pca = PCA(n_components=10)  # init pca object
pcs = pca.fit_transform(stds)  # fit and transform the angles data

# Initialize list that will store Z histograms of all age groups
Zs = []
ratios = []

# Iterating through the list of files for each age group
for age_files in filenames:
    # Load data from this particular age group's files and standardize/scale
    stds = []
    for file in age_files:
        try:
            # Data pre-processing/cleaning (same as above)
            with open(file) as curr_file:
                lines = curr_file.readlines()
                lines = [line.rstrip() for line in lines]
            ind = 0
            for i in range(len(lines)):
                if lines[i][:4] == 'Time':
                    ind = i
                    break
            data = np.loadtxt(file, dtype='float', skiprows=ind + 1)
            if len(data[0]) < 11:
                angle = data
            else:
                time = data[:, 0]
                angle = data[:, 1:]

            # The standardization/scaling step
            scaler = StandardScaler()  # initialize a standardizing object
            stds.append(scaler.fit_transform(angle))  # normalize the data
        except UnicodeDecodeError:
            break    
    stds = np.vstack(stds)  # stack to (n_frames, n_segments) for all data

    # Get the values of the principal component loadings for this age group's data, using the PCA object fit on the overall data earlier
    pcs = pca.transform(stds)


    # Extract the first two PCs and initialize a Gaussian Kernel Density Estimator Object to plot the values
    pc1 = pcs[:, 0]
    pc2 = pcs[:, 1]
    pc12 = np.vstack((pc1, pc2))
    kde = gaussian_kde(pc12)

    # Make the x-y coordinate plane representing the PC1-PC2 space, and calculate the Gaussian KDE values in this space as Z. Store it in the overall list of Zs
    # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    X, Y = np.mgrid[-5:5:100j, -5:5:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)
    Zs.append(Z)

    proj = np.dot(stds, pca.components_.T) #project data onto pcs
    C = np.cov(proj.T) #find covariance of projs
    vars = np.sum(C, axis=1) #sum cov matrix rows or cols doesn't matter bc sym
    ratio = vars / np.sum(vars) #get ratio of vars
    ratios.append(ratio)
        # Define the folder and filename where the .pdf file should be saved
# reports_dir = r"C:\Users\arnav\Documents\wopodyn\reports"
# figname = os.path.join(reports_dir, "N2"+mode+"-overall-trends.pdf")
# pdf = matplotlib.backends.backend_pdf.PdfPages(figname)
# %%
# Initialize the plotting framework
fig, axs = plt.subplots(2, 3, figsize=(10, 5.5))
axr = axs.ravel()
fig.subplots_adjust(right=0.8)

# Iterate through all the age groups and axes in the subplot
for i in range(len(axr)):
    # Normalize the axis bounds over all age groups, as well as the bounds of the Z values (to normalize the colorbars)
    vmin, vmax = np.min(Zs), np.max(Zs)

    # Actually plot the data, and save the .pdf file
    ax = axr[i]
    ax.patch.set_alpha(0.5)
    im = ax.imshow(np.rot90(Zs[i]), cmap='inferno', extent=[-5, 5, -5, 5], vmin=0, vmax=vmax, aspect='equal')
    ax.set(xlim=[-4, 4], ylim=[-4, 4], xlabel='Eigenworm 1', ylabel='Eigenworm 2')
    # plt.colorbar(im1, ax=ax)
    # a = ages[i]
    # ax.set(title=ages[i] + ", n = " + str(len(filenames[i])))
# fig.colorbar(im, ax= axs[:, 2])
# fig.suptitle("N2 " + "PC Histograms over Lifestages", size=20)
cbar_ax = fig.add_axes([0.825, 0.15, 0.025, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, label='Density')
cbar.ax.set_yticks([0, vmax], ['0', 'Max'])
# plt.tight_layout()
# pdf.savefig(fig)
# pdf.close()

# %%
# fig.savefig('reports/figures/devhist.pdf', dpi=300, transparent=True)
# %%
# N2 Swim Eigenworm Histogram
#compile data from all files in directory
# raw_dir = Path('./data/raw')
# suffix = 'N2_Adult'
# path = raw_dir / suffix
# angles = []
# filenames = []
# for count, file in enumerate(path.glob('*')):
#     filenames.append(file)
#     angle = np.loadtxt(file, dtype= 'float', skiprows=1)
#     angles.append(angle[:,1:]) #slice out time column
# print(f'{count+1} files in directory for {suffix}')

# #scale and perform pca on data
# stds = []
# for angle in angles:
#   scaler = StandardScaler() #initialize a standarizing object
#   stds.append(scaler.fit_transform(angle)) #normalize the data
# stds = np.vstack(stds) #stack to (n_frames, n_segments) for all data

# pca = PCA(n_components=10) #init pca object
# pcs = pca.fit_transform(stds) #fit and transform the angles data

# kde = gaussian_kde(pcs[:,0:2].T)
# xmin = np.min(pcs[:,0])
# xmax = np.max(pcs[:,0])
# ymin = np.min(pcs[:,1])
# ymax = np.max(pcs[:,1])

# gmin = np.min([xmin, ymin]) # left bottom min
# gmax = np.max([xmax, ymax]) # right top min

# fmin = np.min([abs(gmin), abs(gmax)]) # min for lim

# X, Y = np.mgrid[gmin:gmax:100j, gmin:gmax:100j]
# positions = np.vstack([X.ravel(), Y.ravel()])
# Z = np.reshape(kde(positions).T, X.shape)

# fig, ax = plt.subplots(figsize=(1.3,1.3))
# im1 = ax.imshow(np.rot90(Z), cmap='viridis',
#           extent=[gmin, gmax, gmin, gmax])
# ax.set(xlim = [-4, 4], 
#         xticks = np.arange(-3, 4),
#         ylim = [-4, 4], 
#         yticks = np.arange(-3, 4),
#         xlabel = 'Eigenworm 1', 
#         ylabel = 'Eigenworm 2')
# fig.colorbar(im1, ax=ax)
# figname = Path('reports/figures/swimhisto.pdf')
# fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")
# %%
# All stages cumultaive variance
cmap = matplotlib.cm.get_cmap('inferno')
fig, ax = plt.subplots(figsize=(4,4))
for i, ratio in enumerate(ratios):
    var = np.hstack(([0], ratio))
    cumvar = np.cumsum(var)
    rgba = cmap(i/len(ratios))
    ax.plot(cumvar, c=rgba, linewidth=2, label=ages[i][1:])
    ax.plot(var, c=rgba, linewidth=2, ls='--')
ax.set( xlim=[0,10],
        xticks = np.arange(11),
        xticklabels = [str(tick) for tick in np.arange(11)],
        xlabel = "Eigenworms",
        ylim=[0, 1.1],
        ylabel = "Cumulative Variance Explained")
ax.legend()
# %%
# N2 Swimming Cumulative Variance
fig, ax = plt.subplots(figsize=(1.3, 1.3))
cumvar = np.cumsum(np.hstack(([0], ratios[-1])))
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
# N2 Swim 10 second kymograph
filename = Path('data/10sec_082522_N2_Adult_swim_0005.txt')
data = np.loadtxt(filename, dtype= 'float', skiprows=1)
angles = data[:,1:]*0.1
fig, ax = plt.subplots(figsize=(4,1.3))

heatmap(data[:,0], angles, fig=fig, ax=ax)
figname = Path('reports/figures/10seckymo.pdf')
fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")

# %%
# N2 Swim 10 Second Eigenworm Loadings
scaler = StandardScaler() #initialize a standarizing object
std = scaler.fit_transform(angles) #normalize the data

pcs = pca.transform(std)

fig, ax = plt.subplots(figsize=(4, 1.3))
ax.plot(pcs[:,:4])
ax.set(xlabel='Time (s)',
        xticks=[0, 140],
        xticklabels=['0', '10'],
        xlim=[0, pcs.shape[0]],
        ylabel='Eigenworm Loadings',
        )

figname = Path('reports/figures/10secload.pdf')
fig.savefig(figname, dpi=300, transparent=True, bbox_inches="tight")

# %%
# N2 Adult Crawl Eigenworm Histogram
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
