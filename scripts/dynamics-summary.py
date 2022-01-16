# %% codecell
#Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns
sns.set_context('poster')
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal
from sklearn.manifold import TSNE

import src.vizualization as viz

data = np.loadtxt(r"C:\Users\Scott\Documents\python\wopodyn\data\raw\cwn2_Adults\070721_cwn-2_A1_Swim_0001_W3.txt", dtype= 'float', skiprows=1)
fig, ax = viz.heatmap(data[:,0], data[:,1:])
