"""Plot heatmap visualization of worm pose in time."""


import numpy as np
import matplotlib.pyplot as plt


def heatmap(time, angles, clip=None, timestep=10):
    """Plot heatmap of pose data."""
    xtimes = np.arange(min(time), max(time), timestep)
    xticks = []
    for i in xtimes:
        ind = np.searchsorted(time, i)
        xticks.append(ind)
        xticklabels = [str(i) for i in xtimes]

    if clip is not None:
        stop_idx = (np.abs(time - clip)).argmin()

    # Set segment ticks and labels
    yticks = np.arange(0, 10)
    yticklabels = reversed([str(i+1) for i in yticks])

    # vs = 1 #hard limit
    vs = max([abs(np.amin(angles)), abs(np.amax(angles))])

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(angles.T, aspect='auto', cmap='seismic', origin='lower',
                   vmin=-vs, vmax=vs)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Time (s)')
    if clip is not None:
        ax.set_xlim([0, stop_idx])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel('Segment')
    fig.colorbar(im, label='Angle (radians)')

    return fig, ax
