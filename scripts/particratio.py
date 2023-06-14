#%%
#Imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.backends.backend_pdf
import seaborn as sns
import pandas as pd
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde, ttest_ind
from pathlib import Path

#%%
raw_dir = Path('./data/raw/crawling')
suffixs = ['N2_L1', 'N2_LateL1', 'N2_L2',  'N2_L3', 'N2_L4', 'N2_Adult']
all_prs = []
for suffix in suffixs:
    print(suffix)
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
    prs = []
    for angle in angles:
        scaler = StandardScaler() #initialize a standarizing object
        std = scaler.fit_transform(angle) #fit and transform the data
        stds.append(std) #normalize the data
        pca = PCA(n_components=10) #init pca object
        pcs = pca.fit_transform(std) #fit and transform the angles data
        evs = pca.explained_variance_
        pr = np.sum(evs)**2/np.sum(evs**2)
        prs.append(pr)

    all_prs.append(prs)
# %%
fig, ax = plt.subplots(figsize=(3,2))
# for i in range(len(all_prs)):
ax = sns.violinplot(data=all_prs, inner='point', saturation=0.6)
ax.set(xticks=np.arange(0,6),
       xticklabels=['young L1', 'late L1', 'L2', 'L3', 'L4', 'Adult'], 
       ylabel='Participation Ratio',
       title='Participation Ratio of Crawling')

# %%
fig.savefig('crawling_prs.pdf', bbox_inches='tight', dpi=300)
# %%
for i in range(len(all_prs)):
    for j in range(len(all_prs)):
        if (i != j) & (i < j):
            print(f'{suffixs[i]} vs {suffixs[j]}')
            t, p = ttest_ind(all_prs[i], all_prs[j], equal_var=True)
            print(f'p = {p}')
# t, pval = ttest_ind(prs_l1, prs_adult, equal_var=False)
# print(f'Means of L1 and Adult PR are unequal with p = {pval} (welch\'s t-test)')
# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.violinplot(data=[prs_l1, prs_adult], inner='point')
ax.set(xticklabels=['L1', 'Adult'], ylabel='Participation Ratio', title=f'Participation Ratio of Swimming L1 and Adult \n (Welch\'s t-test p = {pval:.2e})')
# %%
for pr in all_prs: 
     print(np.mean(pr))
# %%
