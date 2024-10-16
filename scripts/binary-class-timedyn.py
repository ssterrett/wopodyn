# Imports
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from keras import Sequential
from keras.layers import Dense
from src.vizualization import heatmap
import numpy as np
np.random.seed(1)
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.backends.backend_pdf

if __name__ == "__main__":
    pc_num = 10
    # Import Swimming Data
    raw_dir = r"C:\Users\arnav\Documents\data"
    suffix = r"\N2_A1_Swimming"
    path = raw_dir + suffix
    x_swim = np.zeros((1, 10))
    y_train = []
    filenames = []
    for root, dirs, files in os.walk(path):
        for count, file in enumerate(files):
            filenames.append(os.path.join(root, file))
            to_add = np.loadtxt(filenames[count], dtype='float', skiprows=1)
            scaler = StandardScaler()  # initialize a standardizing object
            x_swim = np.vstack((x_swim, scaler.fit_transform(to_add[:, 1:])))  # slice out time column
    x_swim = x_swim[1:, :]

    K = 20
    x_swim_bar = np.zeros((len(x_swim) - K + 1, 1))

    for i in range(K-1, -1, -1):
        stop = len(x_swim) - K + i + 1
        x_swim_bar = np.hstack((x_swim_bar, x_swim[i:stop, :]))
    x_swim = x_swim_bar[:, 1:]

    pca = PCA(n_components=10*K)  # init pca object
    pcs = pca.fit_transform(x_swim)  # fit and transform the angles data
    x_swim = pcs[:, :pc_num]
    y_train = np.ones(len(x_swim))

    # Import Crawling Data
    suffix = r"\N2_A1_Crawling"
    path = raw_dir + suffix
    filenames = []
    x_crawl = np.zeros((1, 10))
    for root, dirs, files in os.walk(path):
        for count, file in enumerate(files):
            filenames.append(os.path.join(root, file))
            to_add = np.loadtxt(filenames[count], dtype='float', skiprows=1)
            scaler = StandardScaler()  # initialize a standardizing object
            x_crawl = np.vstack((x_crawl, scaler.fit_transform(to_add[:, 1:])))  # slice out time column
    # x_crawl = np.random.uniform(np.min(x_crawl), np.max(x_crawl), np.shape(x_crawl))

    x_crawl_bar = np.zeros((len(x_crawl) - K + 1, 1))

    for i in range(K-1, -1, -1):
        stop = len(x_crawl) - K + i + 1
        x_crawl_bar = np.hstack((x_crawl_bar, x_crawl[i:stop, :]))
    x_crawl = x_crawl_bar[:, 1:]

    pca = PCA(n_components=10*K)  # init pca object
    pcs = pca.fit_transform(x_crawl)  # fit and transform the angles data
    x_crawl = pcs[:, :pc_num]
    y_train = np.hstack((y_train, np.zeros(len(x_crawl))))
    x_train = np.vstack((x_swim, x_crawl))

    raw_dir = r"C:\Users\arnav\Documents\data\Transitions"
    # Perform Classification
    standardizer = StandardScaler()
    x_train = standardizer.fit_transform(x_train)
    to_test = np.loadtxt(raw_dir+r"\042922_N2_A1_transition_0004.txt", dtype='float', skiprows=1)
    x_test = to_test[:, 1:]

    x_test_bar = np.zeros((len(x_test) - K + 1, 1))

    for i in range(K - 1, -1, -1):
        stop = len(x_test) - K + i + 1
        x_test_bar = np.hstack((x_test_bar, x_test[i:stop, :]))
    x_test = x_test_bar[:, 1:]

    pca = PCA(n_components=10 * K)  # init pca object
    pcs = pca.fit_transform(x_test)  # fit and transform the angles data
    x_test = pcs[:, :pc_num]

    sw_st = 49.287
    cr_st = 164.643
    y_test = np.concatenate((np.zeros(int(sw_st*14)+1), np.ones(int(14*cr_st-14*sw_st)), np.zeros(len(x_test)-int(14*cr_st))))
    models, accuracy, scores = {}, {}, {}
    # Initialize all model types to test
    # models['Reg'] = LogisticRegression()
    # models['Tree'] = DecisionTreeClassifier()
    # models['Random Forest'] = RandomForestClassifier()
    # models['Naive Bayes'] = GaussianNB()
    models['KNN'] = KNeighborsClassifier()
    # models['Ridge'] = RidgeClassifier()

    print("test " + str(np.sum(y_test)))

    # Fit each model and predict
    for key in models.keys():
        models[key].fit(x_train, y_train)
        preds = models[key].predict(x_test)
        print(key+" "+str(sum(preds)))
        accuracy = accuracy_score(preds, y_test)
        print(accuracy)
        scores[key] = models[key].predict_proba(x_test)
    b_s = len(x_train)//20
    # probs = []
    models['NN'] = Sequential()
    models['NN'].add(Dense(10, activation='relu', kernel_initializer='random_normal', input_dim=10))
    models['NN'].add(Dense(4, activation='relu', kernel_initializer='random_normal'))
    models['NN'].add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    models['NN'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    models['NN'].fit(x_train, y_train, batch_size=b_s, epochs=80, verbose=True)
        # probs.append(models['NN'].evaluate(x_test, y_test, verbose=0))
    # print(probs)
    # Plot probability scores over i/time
    model = 'NN'
    if model == 'NN':
        prob_sw = models['NN'].predict(x_test)
        print(models['NN'].evaluate(x_test, y_test))
    else:
        plot_d = scores[model]
        prob_sw = plot_d[:, 1]
    smooth_prob = []
    k = 40
    for i in range(len(prob_sw)-k):
        r = 0
        for j in range(k):
            r = r + prob_sw[i+j]
        smooth_prob.append(r/k)
    lps = len(prob_sw)
    lsp = len(smooth_prob)
    plt.title(f"Swimming Classification Probability with NN (batch size = {b_s}) and smoothing parameter = {k}")
    plt.plot(np.linspace(0, lps/14, lps), prob_sw, lw=0.5, color='purple', alpha=0.1, label='raw probabilities')
    plt.plot(np.linspace(0, lps/14, lps), np.concatenate((np.zeros(int((lps - lsp) / 2)), smooth_prob, np.zeros(int((lps - lsp) / 2))), axis=None), lw=0.5, label='smoothed probabilities')
    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Classifying Point as Swimming")
    plt.axhline(0.5, 0, int(len(smooth_prob)/14), lw=1, ls='-.', c='k', label='Probability = 0.5')
    plt.axvline(int(sw_st), 0, 1, lw=1, ls='-.', c='g', label='swimming starts')
    plt.axvline(int(cr_st), 0, 1, lw=1, ls='-.', c='r', label='swimming stops')
    plt.ylim([-0.2, 1.2])

    # Put a legend to the right of the current axis
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), prop={'size': 8})
    plt.show()

