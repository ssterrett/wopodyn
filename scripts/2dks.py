# %%
# Tests for 2-dimensional ks statistic (2dks)
# %%
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr
from scipy.stats import genextreme
# %%
# Functions from https://github.com/syrte/ndtest/blob/master/ndtest.py
def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples. 
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.

    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.

    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that the two samples are not significantly different. (cf. Press 2007)

    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8

    '''
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            #ix1 = random.choice(n, n1, replace=True)
            #ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p


def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)


def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d


# %%
# Generate data from two samples and perform KS test
pdf = PdfPages('reports/figures/2dks-tests.pdf')

#hyperparams
sample_size = 1000 #number of samples to draw
sigmas = [0.1, 0.05, 0.01] # ring variance
center_weights = [0.3, 0.1, 0.01] # ratio of points in centroid
center_sigmas = [0.1, 0.05, 0.01] # variance of centroid

for sigma in sigmas:
    for center_weight in center_weights:
        for center_sigma in center_sigmas:
            # Sample 1: ring with variance and no centroid
            phase1 = np.random.uniform(0, 2*np.pi, sample_size)
            x1 = np.sin(phase1) + np.random.normal(0, sigma, (sample_size))
            y1 = np.cos(phase1) + np.random.normal(0, sigma, (sample_size))

            # Sample 2: ring with variance and weighted centroid
            phase2 = np.random.uniform(0, 2*np.pi, int(sample_size*(1-center_weight)))
            x2 = np.sin(phase2) + np.random.normal(0, sigma, int((sample_size*(1-center_weight))))
            x2 = np.hstack((x2,  random.normal(0, center_sigma, int((sample_size*center_weight)))))

            y2 = np.cos(phase2) + np.random.normal(0, sigma, int((sample_size*(1-center_weight))))
            y2 = np.hstack((y2,  random.normal(0, center_sigma, int((sample_size*center_weight)))))

            # 2D KS test
            p = ks2d2s(x1, y1, x2, y2)

            # plotting
            fig, axes = plt.subplots(1, 2, figsize=(12,7), dpi=150)

            ax = axes[0]
            ax.scatter(x1, y1, c='r')
            ax.set(aspect='equal', xlim=[-1.4, 1.4], ylim=[-1.4, 1.4], title='Sample 1')

            ax = axes[1]
            ax.scatter(x2, y2, c='k')
            ax.set(aspect='equal', xlim=[-1.4, 1.4], ylim=[-1.4, 1.4], title='Sample 2')

            fig.suptitle(f'Num. Samples:{sample_size}, Centroid weight: {center_weight}, Centroid variance: {center_sigma}, Ring Variance: {sigma}: p-val = {p:.3f}', size=14)
            fig.tight_layout()

            # save to pdf
            pdf.savefig(fig)
pdf.close()
