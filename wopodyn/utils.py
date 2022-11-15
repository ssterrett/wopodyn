import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def loadwl(filename):
    """ Returns angles and time from wormlab output
    Args:
        filename: relative path of .txt wormlab output
    
    Returns:
        time: array of timestamps in seconds
        angles: array of joint angles in radians
    """
    filename = Path(filename) # convert to pathlib Path
    assert filename.suffix == '.txt', 'Filename not in .txt format'
    
    # Get worm size for unit conversion
    n_segments = 11
    with open(filename) as f:
        first_line = f.readline()
    ls = first_line.split()
    size = int(ls[-2])
    step = (size/1000)/n_segments

    # Load data and extract time (s) and angles (rads)
    data = np.loadtxt(filename, dtype= 'float', skiprows=1)
    time = data[:, 0] # time in seconds
    angles = data[:,1:]*step # joint angles in radians

    return time, angles

def eigenworm(angles):
    """ Standardize and fit a PCA object to angle data.
    
    Args:
        angles: list of joint angle arrays in radians with shape (time, angles)

    Returns:
        pca: sklearn PCA object
        pcs: loadings for all angles concatenated
    """
    assert type(angles) == list, f'Expected list, got {type(angles)} instead'
    stds = []
    for angle in angles:
        scaler = StandardScaler() #initialize a standarizing object
        stds.append(scaler.fit_transform(angle)) #normalize the data
    stds = np.vstack(stds) #stack to (n_frames, n_segments) for all data

    pca = PCA(n_components=10) #init pca object
    pcs = pca.fit_transform(stds) #fit and transform the angles data

    return pca, pcs