# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
# %%
data = np.arange(100)
data = np.reshape(data, (10, 10), 'F')
# %%
def delay_embed(arr, K):
    
    return arr
# %%
def plot_eb(arr):

    return arr
# %%
# delay embed data
embed = []
for i in range(data.shape[1]-(K-1)):
    print(i)
    col = np.reshape(data[:, i:i+K], (-1), 'F')
    embed.append(col)
embed = np.array(embed)
print(embed)
# %%
unembed = np.reshape(embed[0,:], (10,K), order='F')
print(unembed)


# %%
# real data
data = np.loadtxt(r"data/020422_N2_A1_Swim_0011.txt", skiprows=1)
data = data[:,1:].T
plt.imshow(data[:,:100], aspect='auto')
# %%
K=20
embed = []
for i in range(data.shape[1]-(K-1)):
    col = np.reshape(data[:, i:i+K], (-1), 'F')
    embed.append(col)
embed = np.array(embed)

# %%
scaler = StandardScaler()
embed_norm = scaler.fit_transform(embed)

# %%
pca = PCA()
pca.fit_transform(embed_norm)
# %%
eigenbehaviors = pca.components_

# %%
eigenbehavior = eigenbehaviors[0,:]
unembed = np.reshape(eigenbehavior, (10,K), order='F')

# %%
plt.imshow(unembed)
# %%
cumvar = np.cumsum(np.hstack(([0], pca.explained_variance_ratio_)))
plt.plot(cumvar)
plt.xlim([0,10])
# %%
# try crawling
data = np.loadtxt(r"data\raw\N2_Adultcrawl\081520_N2_Adult_Crawl_0001_W1.txt", skiprows=1)
data = data[:,1:].T
plt.imshow(data[:,:100], aspect='auto')

# %%
K=20
embed = []
for i in range(data.shape[1]-(K-1)):
    col = np.reshape(data[:, i:i+K], (-1), 'F')
    embed.append(col)
embed = np.array(embed)

# %%
scaler = StandardScaler()
embed_norm = scaler.fit_transform(embed)

# %%
pca = PCA()
pca.fit_transform(embed_norm)
# %%
eigenbehaviors = pca.components_

# %%
eigenbehavior = eigenbehaviors[0,:]
unembed = np.reshape(eigenbehavior, (10,K), order='F')

# %%
plt.imshow(unembed)
# %%
cumvar = np.cumsum(np.hstack(([0], pca.explained_variance_ratio_)))
plt.plot(cumvar)
plt.xlim([0,10])# %%

# %%
# try transitions
data = np.loadtxt(r"data\050922_N2_A1_transition_0005.txt", skiprows=1)
data = data[:,1:].T
plt.imshow(data, aspect='auto')

# %%
K=25
embed = []
for i in range(data.shape[1]-(K-1)):
    col = np.reshape(data[:, i:i+K], (-1), 'F')
    embed.append(col)
embed = np.array(embed)

# %%
scaler = StandardScaler()
embed_norm = scaler.fit_transform(embed)

# %%
pca = PCA()
pca.fit_transform(embed_norm)
# %%
eigenbehaviors = pca.components_

# %%
fig, axes = plt.subplots(5, 2, figsize=(12,9))
axes = axes.ravel()
for i in range(10):
    eigenbehavior = eigenbehaviors[i,:]
    unembed = np.reshape(eigenbehavior, (10,K), order='F')
    axes[i].imshow(unembed, cmap='bwr')
    axes[i].set(title=f'Eigenbehavior {i}', xlabel='Time (samples)',
                ylabel='Segment Number')
plt.tight_layout()
# %%
cumvar = np.cumsum(np.hstack(([0], pca.explained_variance_ratio_)))
plt.plot(cumvar)
plt.xlim([0,10])
# %%
pcs = pca.fit_transform(embed_norm)
pcs2 = np.square(pcs)
norm = np.sum(pcs2, axis=1)
A_swim = (pcs2[:, 4] + pcs2[:, 5] + pcs2[:,6])/norm
A_crawl = (pcs2[:, 1] + pcs2[:, 2])/norm

# %%
fig, axes = plt.subplots(2, 1, figsize=(12,9))

ax = axes[0]
ax.imshow(data, aspect='auto', cmap='bwr')
ax.set(xticks=[], ylabel='Segment Number')

ax = axes[1]
ax.plot(A_swim, label='$A_{Swim}$')
ax.plot(A_crawl, label='$A_{Crawl}$')
ax.set(xlim=[0, pcs[:,2].shape[0]], xlabel='Time (samples)')
ax.legend(loc='upper center', fontsize='x-large')
fig.tight_layout()

# %%
plt.plot(pcs[:, 1:3])
plt.xlim([0, 200])
# %%
ensemble_crawl = pcs[:,1]**2 + pcs[:,2]**2
ensemble_crawl = (ensemble_crawl - np.mean(ensemble_crawl))/np.var(ensemble_crawl)
ensemble_swim = pcs[:,4]**2 + pcs[:,5]**2
ensemble_swim = (ensemble_swim - np.mean(ensemble_swim))/np.var(ensemble_swim)

plt.plot(ensemble_crawl, label='Crawl')
plt.plot(ensemble_swim, label='Swim')
# %%
ica = FastICA(max_iter=500, random_state=44, whiten=False)
# %%
ics = ica.fit_transform(embed_norm)
# %%
eigenbehaviors = ica.components_
# %%
fig, axes = plt.subplots(5, 2, figsize=(12,9))
axes = axes.ravel()
for i in range(10):
    eigenbehavior = eigenbehaviors[i,:]
    unembed = np.reshape(eigenbehavior, (10,K), order='F')
    axes[i].imshow(unembed)
    axes[i].set(title=f'Eigenbehavior {i}', xlabel='Time (samples)',
                ylabel='Segment Number')
plt.tight_layout()
# %%
plt.plot(ics[:,:5])
plt.ylim([-0.1, 0.1])
# %%
pcs[:,2].shape
# %%
