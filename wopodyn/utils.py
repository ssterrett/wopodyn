import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def loadwl(filename, mode='au'):
    """ Returns angles and time from wormlab output
    Args:
        filename : string
            relative path of .txt wormlab output
        mode : {'au', 'rads'}, default: 'au'
        return data in units of arbitrary or radians
    
    Returns:
        time : array of timestamps in seconds
        angles : array of joint angles in radians
    """
    filename = Path(filename) # convert to pathlib Path
    assert filename.suffix == '.txt', 'Filename not a .txt file'
    
    try:
        # Get worm size for unit conversion
        n_segments = 11
        with open(filename) as f:
            first_line = f.readline()
        ls = first_line.split()
        # size = int(ls[-2])
        # step = (size/1000)/n_segments

    except UnicodeDecodeError:
        print(f'{filename} has unknown character')

    try:
        # Load data and extract time (s) and angles (rads)
        data = np.loadtxt(filename, dtype= 'float', skiprows=1)     
        time = data[:, 0] # time in seconds
        if mode == 'au':
            angles = data[:,1:] # joint angles in au
        # if mode == 'rad':
        #     angles = data[:,1:]*step # joint angles in radians
    except ValueError as e:
        print(f'{filename} has missing rows')


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

    return pca, pcs, stds