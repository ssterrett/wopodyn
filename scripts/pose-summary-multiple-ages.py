# Imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
plt.style.use('seaborn-talk')

if __name__ == "__main__":
    # Note about formatting files and folders:
    # Keep your data in .txt files that are organized in folders for each strain, age, and mode (crawling/swimming)
    # Make sure to name these folders as "strain_age_mode", for example "N2_L1_swimming" or "egl_L3_crawling"
    # Keep all of these folders in a broader subfolder, given by the path in raw_dir
    raw_dir = r"C:\Users\arnav\Documents\data\all_data_no_large"
    strain = r"\N2"
    ages = [r"_L1", r"_lateL1", r"\_L2", r"_L3", r"_L4", r"_A1"]
    mode = r"_crawling"

    # angles is a list where each element is the data from one age group
    # filenames is a list where each element is an array of filenames from one age group
    angles, filenames = [], []

    # Import the data
    for age in ages:
        path = raw_dir + strain + age + mode
        # Initialize the angle and filename element for this particular age group, which will be appended to the overall angles, filenames
        age_angle = []
        age_files = []
        # Loops through all .txt files in the subfolder made for this strain, age, and mode
        for root, dirs, files in os.walk(path):
            for count, file in enumerate(files):
                age_files.append(os.path.join(root, file))
                # Data Pre-processing/cleaning step to remove header of multiple lines (at times)
                with open(age_files[count]) as curr_file:
                    lines = curr_file.readlines()
                    lines = [line.rstrip() for line in lines]
                ind = 0
                for i in range(len(lines)):
                    if lines[i][:4] == 'Time':
                        ind = i
                        break
                # Actually loading the data from .txt
                angle = np.loadtxt(age_files[count], dtype='float', skiprows=ind + 1)

                # A check to make sure that the data is 10 dimensional
                if len(angle[0]) < 11:
                    age_angle.append(angle)
                else:
                    age_angle.append(angle[:, 1:])  # slice out time column
        # Append the age group specific angle/filenames to the overall list
        angles.append(age_angle)
        filenames.append(age_files)

    # standardize/scale the data
    stds = []
    for age_angles in angles:
        for angle in age_angles:
            scaler = StandardScaler()  # initialize a standardizing object
            stds.append(scaler.fit_transform(angle))  # normalize the data
    stds = np.vstack(stds)  # stack to (n_frames, n_segments) for all data

    # Run PCA
    pca = PCA(n_components=10)  # init pca object
    pcs = pca.fit_transform(stds)  # fit and transform the angles data

    # Initialize lists that will store the axis bounds and Z histograms of all age groups
    xmins, xmaxes, ymins, ymaxes, Zs = [], [], [], [], []

    # Iterating through the list of files for each age group
    for age_files in filenames:
        # Load data from this particular age group's files and standardize/scale
        stds = []
        for file in age_files:
            # Data pre-processing/cleaning (same as above)
            with open(file) as curr_file:
                lines = curr_file.readlines()
                lines = [line.rstrip() for line in lines]
            ind = 0
            for i in range(len(lines)):
                if lines[i][:4] == 'Time':
                    ind = i
                    break
            data = np.loadtxt(file, dtype='float', skiprows=ind + 1)
            if len(data[0]) < 11:
                angle = data
            else:
                time = data[:, 0]
                angle = data[:, 1:]

            # The standardization/scaling step
            scaler = StandardScaler()  # initialize a standardizing object
            stds.append(scaler.fit_transform(angle))  # normalize the data
        stds = np.vstack(stds)  # stack to (n_frames, n_segments) for all data

        # Get the values of the principal component loadings for this age group's data, using the PCA object fit on the overall data earlier
        pcs = pca.transform(stds)

        # Extract the first two PCs and initialize a Gaussian Kernel Density Estimator Object to plot the values
        kde1 = pcs[:, 0]
        kde2 = pcs[:, 1]
        kde12 = np.vstack((kde1, kde2))
        kde = gaussian_kde(kde12)

        # Get the bounds of these two loadings, and append them to the overall lists
        xmin = np.min(kde1)
        xmax = np.max(kde1)
        ymin = np.min(kde2)
        ymax = np.max(kde2)

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxes.append(xmax)
        ymaxes.append(ymax)

        # Make the x-y coordinate plane representing the PC1-PC2 space, and calculate the Gaussian KDE values in this space as Z. Store it in the overall list of Zs
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions).T, X.shape)
        Zs.append(Z)

    # Define the folder and filename where the .pdf file should be saved
    # reports_dir = r"C:\Users\arnav\Documents\wopodyn\reports"
    # figname = os.path.join(reports_dir, "N2"+mode+"-overall-trends.pdf")
    # pdf = matplotlib.backends.backend_pdf.PdfPages(figname)

    # Initialize the plotting framework
    fig, axs = plt.subplots(2, 3, figsize=(24, 16))
    axr = axs.ravel()

    # Iterate through all the age groups and axes in the subplot
    for i in range(len(axr)):
        # Normalize the axis bounds over all age groups, as well as the bounds of the Z values (to normalize the colorbars)
        xmin, xmax, ymin, ymax = min(xmins), max(xmaxes), min(ymins), max(ymaxes)
        vmin, vmax = np.min(Zs), np.max(Zs)

        # Actually plot the data, and save the .pdf file
        ax = axr[i]
        ax.patch.set_alpha(0.5)
        im1 = ax.imshow(np.rot90(Zs[i]), cmap='viridis', extent=[min(xmin, ymin), max(xmax, ymax), min(xmin, ymin), max(xmax, ymax)], vmin=vmin, vmax=vmax)
        ax.set(xlim=[min(xmin, ymin), max(xmax, ymax)], ylim=[min(xmin, ymin), max(xmax, ymax)], xlabel=r"$PC_1$", ylabel=r"$PC_2$")
        plt.colorbar(im1, ax=ax)
        a = ages[i]
        ax.set(title=a[4:] + ", n = " + str(len(filenames[i])))
        fig.suptitle("N2 " + mode[1:] + " PC Histograms over Lifestages", size=20)

    # pdf.savefig(fig)
    # pdf.close()




