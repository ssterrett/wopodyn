# Imports
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.vizualization import heatmap
import numpy as np
np.random.seed(1)
import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA

# # Delay embed a dataset 'angles' with a delay factor/k-value of K
# def delay_embed(angles, K):
#     x_bar = np.zeros((len(angles) - K + 1, 1))
#     for i in range(K-1, -1, -1):
#         stop = len(angles) - K + i + 1
#         x_bar = np.hstack((x_bar, angles[i:stop, :]))
#     x_bar = x_bar[:, 1:]
#     return x_bar

# # Draw scree plot of a PCA object onto a .pdf
# def draw_scree(pca_obj, pdf_to_draw):
#     fig, ax = plt.subplots(figsize=(24, 16))
#     ax.set_title(f"Scree Plot for K = {i}")
#     ax.plot(np.cumsum(np.hstack(([0], pca_obj.explained_variance_ratio_))))
#     setpoints = np.arange(0, (i*3)//2, i//5)
#     ax.vlines(setpoints, 0, 1, 'red', 'dotted', lw=0.2)
#     ax.set(xlim=[0, i*10+1], xticks = np.arange(i*10+1, step=i*2), xlabel = "Num Components", ylabel = "Cumulative Variance Explained")
#     pdf_to_draw.savefig(fig)

# # Draw first 10 PCs of a PCA object corresponding  onto a .pdf
# def draw_pcs(angles, pca_obj, pdf_to_draw):
#     angle = angles[:, 1:]
#     pcs = pca_obj.transform(StandardScaler().fit_transform(delay_embed(angle, i)))
#     l = len(pcs)
#     time = angles[:l, 0]
#     fig, ax = plt.subplots(figsize=(24, 16))
#     ax.set_title(f"PCs over time for K = {i}")
#     lines = ax.plot(time, pcs[:, :10], lw=0.002)
#     ax.set(xlim=[min(time), max(time)], xlabel="Time")
#     ax.legend(lines, ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'))
#     pdf_to_draw.savefig(fig)

if __name__ == "__main__":
    # Import Data
    path = r"C:\Users\arnav\Documents\data\Transitions"

    x = np.zeros((1, 10))
    filenames = []
    # filename = os.path.join(reports_dir, f"all-heatmaps.pdf")
    # pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    for root, dirs, files in os.walk(path):
        for count, file in enumerate(files):
            filenames.append(os.path.join(root, file))
            to_add = np.loadtxt(filenames[count], dtype='float', skiprows=1)
            fig, ax = heatmap(to_add[:, 0], to_add[:, 1:], title=filenames[count])
            plt.show()
            x = np.vstack((x, StandardScaler().fit_transform(to_add[:, 1:])))
            # pdf.savefig(fig)
    # pdf.close()
    x = x[1:, :]

    # # Define the folder and filename where the .pdf file should be saved
    # reports_dir = r"C:\Users\arnav\Documents\wopodyn-main\reports"

    # For different k values for delay embedding, draw the scree plot and the first 10 PCs for each file
    # for i in [10, 15, 20, 25]:
    #     filename = os.path.join(reports_dir, f"data-exploration-varyingK-{i}.pdf")
    #     pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    #     pca = PCA(n_components=10*i)  # init pca object
    #     pca.fit(delay_embed(x, i))  # fit the angles data
    #     # draw_scree(pca, pdf)
    #     for file in filenames:
    #         data = np.loadtxt(file, dtype='float', skiprows=1)
    #         draw_pcs(data, pca, pdf)
    #     pdf.close()
