# Imports
import numpy as np
np.random.seed(0)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import scipy.stats as ss


def classify(angles, labels, clsfr, file=None):
    x_train, x_test, y_train, y_test = train_test_split(angles, labels, random_state = 0)
    clsfr.fit(x_train, y_train)
    y_hat = clsfr.predict(x_test)
    print(f"Overall accuracy for {clsfr}: {accuracy_score(y_test, y_hat)}")    
    accs = []
    if file is not None:
        start, end = data_pieces[file]
        x_train_c, x_test_curr, y_train_c, y_test_curr = train_test_split(angles[start:end], labels[start:end], random_state = 0)
        y_hat_curr = clsfr.predict(x_test_curr)
        print(f"{clsfr} accuracy for {file}: {accuracy_score(y_test_curr, y_hat_curr)}")
        accs.append(accuracy_score(y_test_curr, y_hat_curr))
    accs.append(accuracy_score(y_test, y_hat))

def delay_embed(angles, K):
    x_bar = np.zeros((len(angles) - K + 1, 1))
    for i in range(K-1, -1, -1):
        stop = len(angles) - K + i + 1
        x_bar = np.hstack((x_bar, angles[i:stop, :]))
    x_bar = x_bar[:, 1:]
    return x_bar

if __name__ == "__main__":
    path = r"C:\Users\arnav\Documents\data\Transitions"
    timestamp_path = r"C:\Users\arnav\Documents\wopodyn"
    # Import timestamps of human-observed behavior transitions
    timestamps_keys = np.loadtxt(timestamp_path+r'\transition-times.txt', dtype='str', delimiter=',', usecols=0)
    timestamps_values = np.loadtxt(timestamp_path+r'\transition-times.txt', delimiter=',', dtype='float', usecols=range(1, 5))

    timestamps = {path+timestamps_keys[k]: timestamps_values[k] for k in range(len(timestamps_keys))}  

    # Import Data
    i = 10
    x, y, filenames, data_pieces, k = np.zeros((1, 10*i)), np.zeros((1,)), [], {}, 0        
    for root, dirs, files in os.walk(path):
        for count, file in enumerate(files):                                                                                                    #iterate through all files in the path folder
            curr_file = os.path.join(root, file)
            filenames.append(curr_file)                                                                                                         #keep running log of files
            to_add = np.loadtxt(curr_file, dtype='float', skiprows=1)                                                                           #import file data
            cr_e, sw_s, sw_e, cr_s = timestamps[curr_file][0], timestamps[curr_file][1], timestamps[curr_file][2], timestamps[curr_file][3]     #import behavior switch timestamps for current file
            y_temp = []
            for c in range(len(to_add)):                                                                                                        #assign behavior label to each time point in current file data
                if c < cr_e*14: y_temp.append(0)
                elif c < sw_s*14: y_temp.append(1)
                elif c < sw_e*14: y_temp.append(2)
                elif c < cr_s*14: y_temp.append(3)
                else: y_temp.append(0)
            x_curr = delay_embed(StandardScaler().fit_transform(to_add[:, 1:]), i)
            x = np.vstack((x, x_curr))                                                                                                          #delay embed file data and add it to overall X dataset
            y = np.hstack((y, np.array([ss.mode(y_temp[c:c+i-1], axis=None)[0] for c in range(len(y_temp)-i+1)])))                                              #function to transform behavior labels onto delay embedded dataset
            data_pieces[curr_file] = (k, k+len(x_curr))
            k = k+len(x_curr)
    x, y = x[1:, :], y[1:]                                                                                                                                #skip out first row of dataset (set to row of zeros during instantiation)
    
    # Define the folder and filename where the .pdf file should be saved
    reports_dir = r"C:\Users\arnav\Documents\wopodyn-main\reports\trials\file-specific"
    # corrs = np.corrcoef(np.vstack((x.T, [y,])))
    # plt.matshow(corrs)
    # plt.show()

    # pca = PCA(n_components=10*i)                                                                                                                #init pca object
    # elbow_i = i*3//5
    # x_red = pca.fit_transform(x)[:, :elbow_i]                                                                                                          #fit and transform the angles data

    # corrs = np.corrcoef(np.vstack((x_red.T, [y,])))
    # plt.matshow(corrs)
    # plt.show()

    classifiers = [KNeighborsClassifier(), RandomForestClassifier(), MLPClassifier()]

    for classifier in classifiers:
        classify(x, y, classifier, filenames[0])
