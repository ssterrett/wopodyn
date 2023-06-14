#%%
#Imports
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.backends.backend_pdf
import matplotlib
# plt.style.use('seaborn-talk')
# style specs
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
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde, binned_statistic, binned_statistic_2d
from scipy.signal import hilbert


from wopodyn.vizualization import heatmap

import pickle

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
    pca = PCA(n_components=10) #init pca object
    pcs = pca.fit_transform(std) #fit and transform the angles data
    evs = pca.explained_variance_

stds = np.vstack(stds) #stack to (n_frames, n_segments) for all data
pca = PCA(n_components=10) #init pca object
pcs = pca.fit_transform(stds) #fit and transform the angles data
# %%
file = open('pca.obj', 'rb')
pca = pickle.load(file)

# %%
# calculate angular velocity and visualize
sr = 10 #sampling rate in Hz
all_w = np.array([]).reshape(0,)
all_om = np.array([]).reshape(0,)
all_pcs = np.array([]).reshape(0,10)
all_dpcs = np.array([]).reshape(0,10)

for angle in angles:
    scaler = StandardScaler() #initialize a standarizing object
    std = scaler.fit_transform(angle)
    pcs = pca.transform(std)
    dpcs = np.diff(pcs, axis=0)
    al1 = np.angle(hilbert(pcs[:,0]),deg=False)
    w = np.diff(al1)*sr/(2*np.pi) #angular velocity (hz)
    all_om = np.concatenate((all_om, al1[1:]))
    all_w = np.concatenate((all_w, w))
    all_pcs = np.concatenate((all_pcs, pcs[1:, :]))
    all_dpcs = np.concatenate((all_dpcs, dpcs[:, :]))

speed = np.sqrt(np.sum(all_dpcs**2, axis=1))

# # plot mean angular velocity as a function of angle
# fig, ax = plt.subplots(figsize=(12,9))
# n_edges = 5
# edges = np.linspace(-np.pi, np.pi, n_edges)
# for i in range(n_edges-1):
#     inds = (all_om>edges[i])*(all_om<edges[i+1])
#     mean = np.mean(speed[inds])
#     std = np.std(speed[inds])
#     om = (edges[i] + edges[i+1])/2
#     ax.scatter(om, mean, c='k')
#     ax.errorbar(om, mean, yerr=std, c='k')
# ax.set(xlabel='Phase (rad)', 
#        ylabel='Speed (au/s)', 
#        title='N2 Adult Crawling Speed vs. Phase')

# %%
n_edges = 50
mean_speed_crawl = np.zeros(n_edges-1)
oms = np.zeros(n_edges-1)
edges = np.linspace(-np.pi, np.pi, n_edges)
for i in range(n_edges-1):
    inds = (all_om>edges[i])*(all_om<edges[i+1])
    mean = np.mean(speed[inds])
    mean_speed_crawl[i] = mean
    std = np.std(speed[inds])
    om = (edges[i] + edges[i+1])/2
    oms[i] = om
# %%
scipy
# %%
fig, ax = plt.subplots(figsize=(1.2, 1.2), subplot_kw={'projection': 'polar'})
ax.scatter(all_om[:5000]+np.pi/4, speed[:5000], c='navy', s=10, alpha=0.009)
ax.plot(np.append(oms, oms[0])+np.pi/4, np.append(mean_speed_crawl, mean_speed_crawl[0]), c='navy', linewidth=3)
# ax.plot(edges[1:], mean_speed_crawl, c='orange', linewidth=3)
plt.rgrids(radii=[1, 2, 3], labels=[])
ax.set_rmax(4)
ax.tick_params(axis='both', pad=0.1)
ax.set_axisbelow(False)
plt.grid(linewidth=0.5)
# %%
fig.savefig('swimming_speeds.pdf', bbox_inches='tight', dpi=300)

# %%
n_edges = 20
mean_speed = np.zeros(n_edges-1)
edges = np.linspace(-np.pi, np.pi, n_edges)
# histogram of speed by phase
dens, _ = np.histogram(speed, bins=edges)
for i in range(n_edges-1):
    inds = (all_om>edges[i])*(all_om<edges[i+1])
    mean = np.mean(speed[inds])
    mean_speed[i] = mean
    std = np.std(speed[inds])
    om = (edges[i] + edges[i+1])/2
# %%
count , _, _ = binned_statistic(all_om, speed, bins=edges, statistic='count')
count = count/np.sum(count)
# %%
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})
ax.plot(np.append(oms, oms[0]), np.append(count, count[0]), c='sandybrown', linewidth=5)
# ax.plot(np.append(oms, oms[0]), np.append(count_swim, count_swim[0]), c='cornflowerblue', linewidth=5)

# %%
count_swim = count
# %%
#transform session
filename = r"C:\Users\Scott\Documents\python\wopodyn\data\N2_Adult.txt"
data = np.loadtxt(filename, dtype= 'float', skiprows=1)
# start = 0
# stop = 10
# start_idx = (np.abs(data[:,0] - start)).argmin()
# stop_idx = (np.abs(data[:,0] - stop)).argmin()
# time = data[start_idx:stop_idx,0]; angle = data[start_idx:stop_idx,1:]
time = data[:,0]
angle = data[:,1:]
scaler = StandardScaler() #initialize a standarizing object
std = scaler.fit_transform(angle)
pcs = pca.transform(std)
al1 = np.angle(hilbert(pcs[:,0]),deg=False)

fig, axes = plt.subplots(3, 1, figsize=(12,9))
axes = axes.ravel()
ax = axes[0]
ax.imshow(angle.T, aspect='auto')

ax = axes[1]
ax.plot(pcs[:,:1])

ax = axes[2]
ax.plot(al1)
# %%
# calculate angular velocity and visualize
sr = 10 #sampling rate in Hz
all_w = np.array([]).reshape(0,)
all_om = np.array([]).reshape(0,)
all_pcs = np.array([]).reshape(0,10)
all_dpcs = np.array([]).reshape(0,10)

fig, ax = plt.subplots(figsize=(12,9))
for angle in angles:
    scaler = StandardScaler() #initialize a standarizing object
    std = scaler.fit_transform(angle)
    pcs = pca.transform(std)
    dpcs = np.diff(pcs, axis=0)
    al1 = np.angle(hilbert(pcs[:,0]),deg=False)
    w = np.diff(al1)*sr/(2*np.pi) #angular velocity (hz)
    ax.scatter(al1[1:], w, c='k', alpha=0.1)
    all_om = np.concatenate((all_om, al1[1:]))
    all_w = np.concatenate((all_w, w))
    all_pcs = np.concatenate((all_pcs, pcs[1:, :]))
    all_dpcs = np.concatenate((all_dpcs, dpcs[:, :]))

ax.set(xlabel='Angle (radians)', xticks=[-np.pi, 0, np.pi],
        xticklabels=['$-\pi$', '0', '$\pi$'], ylabel='Angular Velocity (Hz)',
        ylim=[0,3], title='Swimming N2 Adults')

fig, ax = plt.subplots(figsize=(12,9))
n, x, _ = plt.hist(all_w, bins=np.linspace(0, 2, 30))
bin_centers = 0.5*(x[1:]+x[:-1])
ax.plot(bin_centers, n, 'o-', c='k')
ax.set(xlabel='Angular Velocity (Hz)', ylabel='Count', title='N2 Adult Swimming')
# %%
speed = np.sqrt(np.sum(all_dpcs**2, axis=1))

# %%

# %%
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
ax.set(xlim = [-5, 5], ylim = [-5, 5], xlabel = r"$PC_1$", ylabel = r"$PC_2$")
# %%
fig, ax = plt.subplots(figsize=(9,9))
ax.hist2d(all_pcs[:,0], all_pcs[:,1], bins=100, cmap='viridis')
ax.set(aspect='equal',
       xlim=[-5, 5],
       ylim=[-5, 5],
       xlabel=r"$PC_1$",
       ylabel=r"$PC_2$",
       title='N2 Adult Swimming Histogram')
# %%
all_w_clip = all_w[(all_w>0.1)*(all_w<2)]
all_pcs_clip = all_pcs[(all_w>0.1)*(all_w<2), :]
fig, ax = plt.subplots(figsize=(9,9))
im = ax.scatter(all_pcs_clip[:,0], all_pcs_clip[:,1], c=all_w_clip, vmin=0, vmax=2, alpha=0.03)
ax.set(aspect='equal',
       xlim=[-5, 5],
       ylim=[-5, 5],
       xlabel=r"$PC_1$",
       ylabel=r"$PC_2$",
       title='N2 Adult Swimming Angular Speed')
cb = fig.colorbar(im)
cb.solids.set(alpha=1)

# %%
fig, ax = plt.subplots(figsize=(9,9))
im = ax.scatter(all_pcs[:,0], all_pcs[:,1], c=speed, vmin=0, vmax=2, alpha=0.03)
ax.set(aspect='equal',
       xlim=[-5, 5],
       ylim=[-5, 5],
       xlabel=r"$PC_1$",
       ylabel=r"$PC_2$",
       title='N2 Adult Swimming Speed')
cb = fig.colorbar(im, label='Speed (au/s)')
cb.solids.set(alpha=1)

# %%
count, _, _, _ = binned_statistic_2d(all_pcs[:,0], all_pcs[:,1], speed, bins=50, range=[[-5, 5], [-5, 5]], statistic='count')
means, _, _, _ = binned_statistic_2d(all_pcs[:,0], all_pcs[:,1], speed, bins=50, range=[[-5, 5], [-5, 5]])

for i in range(count.shape[0]):
    for j in range(count.shape[1]):
        if count[i,j] < 5:
            means[i,j] = np.nan

# %%
fig, ax = plt.subplots(figsize=(9,9))
im = ax.imshow(means, vmin=0, vmax=1.5, cmap='viridis')
ax.set(xticks=[], 
       yticks=[], 
       xlabel='Eigenworm 1',
       ylabel='Eigenworm 2',
       title='N2 Adult Crawling Average Speed')
fig.colorbar(im, label='Speed (au/s)')
# %%
# plot mean angular velocity as a function of angle
fig, ax = plt.subplots(figsize=(12,9))
n_edges = 5
edges = np.linspace(-np.pi, np.pi, n_edges)
for i in range(n_edges-1):
    inds = (all_om>edges[i])*(all_om<edges[i+1])
    mean = np.mean(speed[inds])
    std = np.std(speed[inds])
    om = (edges[i] + edges[i+1])/2
    ax.scatter(om, mean, c='k')
    ax.errorbar(om, mean, yerr=std, c='k')
ax.set(xlabel='Phase (rad)', 
       ylabel='Speed (au/s)', 
       title='N2 Adult Crawling Speed vs. Phase')
# %%
# Rolling Pearson Corrleation
# Set window size to compute moving window synchrony.
r_window_size = 20
# Interpolate missing data.
df = pd.DataFrame(pcs)
df[1]
# Compute rolling window synchrony
rolling_r = df[0].rolling(window=r_window_size, center=True).corr(df[1])
rolling_r.plot()

# %%
# Phase Synchrony
al1 = np.angle(hilbert(pcs[:,0]),deg=False)
al2 = np.angle(hilbert(pcs[:,1]),deg=False)
phase_synchrony = 1-np.sin(np.abs(al1-al2)/2)
plt.plot(al1)
plt.plot(al2)

# %%
# Rolling window time lagged cross correlation

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))
df = pd.DataFrame(pcs)
window_size = 20 #samples
t_start = 0
t_end = t_start + window_size
step_size = 5
rss=[]
while t_end < df.shape[0]:
    d1 = df[0].iloc[t_start:t_end]
    d2 = df[1].iloc[t_start:t_end]
    rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-10,10)]
    rss.append(rs)
    t_start = t_start + step_size
    t_end = t_end + step_size
rss = pd.DataFrame(rss)

f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(rss,cmap='RdBu_r',ax=ax)

heatmap(time, angle)
max_corr = np.max(rss, axis=1)
plt.plot(max_corr)
# %%
filename = r"C:\Users\Scott\Documents\python\wopodyn\data\raw\N2_L1\011222_N2_L1_Swim_0012_W1.txt"
data = np.loadtxt(filename, dtype= 'float', skiprows=1)
time = data[:,0]
angle = data[:,1:]
scaler = StandardScaler() #initialize a standarizing object
std = scaler.fit_transform(angle)
pcs = pca.transform(std)

heatmap(time, angle)

df = pd.DataFrame(pcs)
window_size = 20 #samples
t_start = 0
t_end = t_start + window_size
step_size = 5
rss=[]
while t_end < df.shape[0]:
    d1 = df[0].iloc[t_start:t_end]
    d2 = df[1].iloc[t_start:t_end]
    rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-10,10)]
    rss.append(rs)
    t_start += step_size
    t_end += step_size
rss = pd.DataFrame(rss)

f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(rss,cmap='RdBu_r',ax=ax)

max_corr = np.max(rss, axis=1)
plt.plot(max_corr)
