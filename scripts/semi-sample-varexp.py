# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
# Load Data
fn = 'data/raw/N2_Adult/091321_N2_Adult_Swim_0002_W5.txt'
with open(fn) as f:
    first_line = f.readline()
ls = first_line.split()
size = int(ls[-2])

data = np.loadtxt(fn, dtype='float', skiprows=1)
time = data[:,0]
angles = data[:,1:]

# %%
# Compute both methods and compare

# PCA on raw data
scaler = StandardScaler()
std = scaler.fit_transform(angles)

pca = PCA()
pcs = pca.fit_transform(std)
cumvar = np.cumsum(np.hstack(([0], pca.explained_variance_ratio_)))

fig, ax = plt.subplots(figsize=(12,9))
ax.plot(cumvar, label='PCA Raw') #plot PCA method

# Sum of cov of projections
proj = np.dot(std, pca.components_.T) #project data onto pcs
C = np.cov(proj.T) #find covariance of projs
vars = np.sum(C, axis=1) #sum cov matrix rows or cols doesn't matter bc sym
ratio = vars / np.sum(vars) #get ratio of vars
ax.plot(np.cumsum(np.hstack(([0], ratio))), label='Cov Proj')
ax.set(xlabel='PC', ylabel='Cumulative Variance Explained')
ax.legend()

# Lines are the same - hooray!
