# Imports
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from keras import Sequential
from keras.layers import Dense
from src.vizualization import heatmap
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(1)

plt.style.use('seaborn-v0_8-talk')

if __name__ == "__main__":
    # Import Swimming Data
    raw_dir = r"C:\Users\arnav\Documents\data"
    suffix = r"\N2_A1_Swimming"
    path = raw_dir + suffix
    x_swim = np.zeros((1, 10))
    y_train = []
    filenames_swim = []
    for root, dirs, files in os.walk(path):
        for count, file in enumerate(files):
            filenames_swim.append(os.path.join(root, file))
            to_add = np.loadtxt(filenames_swim[count], dtype='float', skiprows=1)
            scaler = StandardScaler()
            x_swim = np.vstack((x_swim, scaler.fit_transform(to_add[:, 1:])))  # slice out time column
    x_swim = x_swim[1:, :]

    sh = np.shape(x_swim)

    K = 20
    pc_num = K*10
    x_swim_bar = np.zeros((len(x_swim) - K + 1, 1))

    for i in range(K - 1, -1, -1):
        stop = len(x_swim) - K + i + 1
        x_swim_bar = np.hstack((x_swim_bar, x_swim[i:stop, :]))
    x_swim = x_swim_bar[:, 1:]
    #
    # pca = PCA(n_components=10 * K)  # init pca object
    # pcs = pca.fit_transform(x_swim)  # fit and transform the angles data
    # x_swim = pcs[:, :pc_num]
    y_train = np.ones((len(x_swim), 1))

    # Import Crawling Data
    suffix = r"\N2_A1_Crawling"
    path = raw_dir + suffix
    filenames_crawl = []
    x_crawl = np.random.uniform(low=np.min(x_swim), high=np.max(x_swim), size=sh)
    x_crawl_bar = np.zeros((len(x_crawl) - K + 1, 1))

    for i in range(K - 1, -1, -1):
        stop = len(x_crawl) - K + i + 1
        x_crawl_bar = np.hstack((x_crawl_bar, x_crawl[i:stop, :]))
    x_crawl = x_crawl_bar[:, 1:]

    x_train = np.vstack((x_swim, x_crawl))
    y_train = np.vstack((y_train, np.zeros((len(x_crawl), 1))))

    x_train = StandardScaler().fit_transform(x_train)
    x_train = MinMaxScaler().fit_transform(x_train)
    X = np.hstack((x_train, y_train))
    np.random.shuffle(X)
    y_train = X[:, -1]
    x_train = X[:, :-1]

    pca = PCA(n_components=10 * K)  # init pca object
    pcs = pca.fit_transform(x_train)  # fit and transform the angles data
    x_train = pcs[:, :pc_num]

    # cumvar = np.cumsum(np.hstack(([0], pca.explained_variance_ratio_)))
    # print(cumvar)
    # fig = plt.figure(figsize=(16,10))
    # ax = fig.gca()
    # ax.plot(cumvar)
    #
    # ax.set(xlabel="Num Components", ylabel="Cumulative Variance Explained")
    # ax.xaxis.label.set_size(11)
    # plt.show()

    # pc_indices = range(11)
    # legend_text = ()
    # for index in pc_indices:
    #     legend_text = legend_text + (r"$PC_{" + str(index+1) + "}$",)
    #
    # file = np.random.choice(filenames_swim, 1)
    # data = np.loadtxt(file[0], dtype='float', skiprows=1)
    # angle = data[:250, 1:]
    # scaler = StandardScaler()
    # angle_norm = scaler.fit_transform(angle)
    # K = 20
    # s = len(angle) - K + 1
    # time = data[:s, 0]
    # bar = np.zeros((len(angle_norm) - K + 1, 1))
    #
    # for i in range(K - 1, -1, -1):
    #     stop = len(angle_norm) - K + i + 1
    #     bar = np.hstack((bar, angle_norm[i:stop, :]))
    # angle_norm = bar[:, 1:]
    # pcs = pca.transform(angle_norm)
    #
    # total = pcs[:, 0]
    # for index in pc_indices:
    #     total = np.vstack((total, pcs[:, index]))
    #
    # fig, axs = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    # axs = axs.ravel()
    # ax = axs[0]
    # heatmap(time, angle, fig=fig, ax=ax)
    #
    # ax = axs[1]
    # lines = ax.plot(time, total.T, lw=1)
    # ax.set(xlim=[min(time), max(time)])
    # ax.legend(lines, legend_text)
    # plt.show()

    # Perform Classification

    suffix = r"\Transitions"
    path = raw_dir + suffix
    filenames_transition = []
    x_crawl = np.zeros((1, 10))
    for root, dirs, files in os.walk(path):
        for count, file in enumerate(files):
            filenames_transition.append(os.path.join(root, file))

    file_index = 0
    transition_file = filenames_transition[file_index]
    to_test = np.loadtxt(transition_file, dtype='float', skiprows=1)
    x_test = to_test[:, 1:]

    norm = StandardScaler().fit_transform(x_test)
    norm = MinMaxScaler().fit_transform(norm)

    x_test_bar = np.zeros((len(x_test) - K + 1, 1))

    for i in range(K - 1, -1, -1):
        stop = len(x_test) - K + i + 1
        x_test_bar = np.hstack((x_test_bar, norm[i:stop, :]))
    x_test = x_test_bar[:, 1:]
    pcs = pca.transform(x_test)
    x_test = pcs[:, :pc_num]

    cera_times = np.array([[29.357, 34.786, 154.215, 169.429], [11.928, 13.5, 660.5, 697.857], [49.642, 51.571, 289.357, 321.357], [20.142, 23, 183.642, 201.928]])
    brennan_times = np.array([[29.357, 35.143, 154.286, 171.786], [11.928, 14.071, 660.785, 696.642], [49.642, 52.357, 290.857, 324.214], [20.142, 24.428, 184.071, 205.071]])
    kai_times = np.array([[30.643, 38.643, 155.143, 161.715], [12.214, 15.5, 662.928, 683.928], [49.714, 53.5, 300.214, 317.571], [20.142, 24.785, 187.785, 201.071]])
    angie_times = np.array([[29.357, 35.572, 153.786, 171.715], [11.928, 13.714, 659.642, 688.071], [49.571, 52, 288.857, 321.714], [20.142, 24.142, 182, 200.571]])
    mean_times = np.mean(np.array([cera_times, brennan_times, kai_times, angie_times]), axis=0)
    std_times = np.std(np.array([cera_times, brennan_times, kai_times, angie_times]), axis=0)

    ent_water = mean_times[file_index][0]
    sw_st = mean_times[file_index][1]
    cr_st = mean_times[file_index][2]
    dry_time = mean_times[file_index][3]
    y_test = np.concatenate((np.zeros(int(sw_st*14)+1), np.ones(int(14*cr_st-14*sw_st))))
    y_test = np.concatenate((y_test, np.zeros(int(len(x_test)-len(y_test)))))
    models, accuracy, scores = {}, {}, {}

    b_s = len(x_train)//10
    # probs = []
    models['NN'] = Sequential()
    models['NN'].add(Dense(10, activation='relu', kernel_initializer='random_normal', input_dim=pc_num))
    models['NN'].add(Dense(4, activation='relu', kernel_initializer='random_normal'))
    models['NN'].add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    models['NN'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    models['NN'].fit(x_train, y_train, batch_size=b_s*2, epochs=80, verbose=False)

    prob_sw = models['NN'].predict(x_test)
    acc = models['NN'].evaluate(x_test, y_test, verbose=0)
    print(acc)

    smooth_prob = []
    k = 40
    for i in range(len(prob_sw)-k):
        r = 0
        for j in range(k):
            r = r + prob_sw[i+j]
        smooth_prob.append(r/k)

    lps = len(prob_sw)
    lsp = len(smooth_prob)

    plt.title(transition_file[1:-4] + f" Binary Classification Probability")
    plt.plot(np.linspace(0, lps / 14, lps), prob_sw, lw=0.5, color='purple', alpha=0.1, label='raw probabilities')
    plt.plot(np.linspace(0, lps / 14, lps),
             np.concatenate((np.zeros(int((lps - lsp) / 2)), smooth_prob, np.zeros(int((lps - lsp) / 2))), axis=None),
             lw=0.5, label='smoothed probabilities')
    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Classifying Point as Swimming")
    plt.axhline(0.5, 0, int(len(smooth_prob) / 14), lw=1, ls='-.', c='k', label='Probability = 0.5')
    plt.ylim([-0.2, 1.2])

    # Put a legend to the right of the current axis
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), prop={'size': 8})
    plt.show()
    plt.title(transition_file[1:-4] + f" Binary Classification Probability. Accuracy: " + str(acc[1]))
    plt.plot(np.linspace(0, lps / 14, lps), prob_sw, lw=0.5, color='purple', alpha=0.1, label='raw probabilities')
    plt.plot(np.linspace(0, lps / 14, lps),
             np.concatenate((np.zeros(int((lps - lsp) / 2)), smooth_prob, np.zeros(int((lps - lsp) / 2))), axis=None),
             lw=0.5, label='smoothed probabilities')
    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Classifying Point as Swimming")
    plt.axhline(0.5, 0, int(len(smooth_prob) / 14), lw=1, ls='-.', c='k', label='Probability = 0.5')
    plt.fill_between(np.linspace(0, lps / 14, lps), -0.2, 1.2,
                     where=(np.linspace(0, lps / 14, lps) <= sw_st + std_times[file_index][1]) & (
                             np.linspace(0, lps / 14, lps) >= ent_water - std_times[file_index][0]), color='g',
                     alpha=0.3)
    plt.axvline(ent_water, 0, 1, lw=1, ls='-.', c='g', label='enters water')
    plt.axvline(sw_st, 0, 1, lw=1, ls='-.', c='g', label='swimming starts')
    plt.axvline(cr_st, 0, 1, lw=1, ls='-.', c='r', label='swimming stops')
    plt.axvline(dry_time, 0, 1, lw=1, ls='-.', c='r', label='bubble dries')
    plt.fill_between(np.linspace(0, lps / 14, lps), -0.2, 1.2,
                     where=(np.linspace(0, lps / 14, lps) <= dry_time + std_times[file_index][3]) & (
                             np.linspace(0, lps / 14, lps) >= cr_st - std_times[file_index][2]), color='r',
                     alpha=0.3)
    plt.ylim([-0.2, 1.2])

    # Put a legend to the right of the current axis
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), prop={'size': 8})
    plt.show()


    plt.title(transition_file[1:-4] + f" Binary Classification Probability")
    plt.plot(np.linspace(0, lps / 14, lps), prob_sw, lw=0.5, color='purple', alpha=0.1, label='raw probabilities')
    plt.plot(np.linspace(0, lps / 14, lps),
             np.concatenate((np.zeros(int((lps - lsp) / 2)), smooth_prob, np.zeros(int((lps - lsp) / 2))), axis=None),
             lw=0.5, label='smoothed probabilities')
    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Classifying Point as Swimming")
    plt.axhline(0.5, 0, int(len(smooth_prob) / 14), lw=1, ls='-.', c='k', label='Probability = 0.5')
    plt.ylim([-0.2, 1.2])
    plt.xlim([0, sw_st * 3])

    # Put a legend to the right of the current axis
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), prop={'size': 8})
    plt.show()

    plt.title(transition_file[1:-4] + f" Binary Classification Probability")
    plt.plot(np.linspace(0, lps / 14, lps), prob_sw, lw=0.5, color='purple', alpha=0.1, label='raw probabilities')
    plt.plot(np.linspace(0, lps / 14, lps),
             np.concatenate((np.zeros(int((lps - lsp) / 2)), smooth_prob, np.zeros(int((lps - lsp) / 2))), axis=None),
             lw=0.5, label='smoothed probabilities')
    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Classifying Point as Swimming")
    plt.axhline(0.5, 0, int(len(smooth_prob) / 14), lw=1, ls='-.', c='k', label='Probability = 0.5')
    plt.ylim([-0.2, 1.2])
    plt.xlim([cr_st * 0.8, dry_time * 1.2])

    # Put a legend to the right of the current axis
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), prop={'size': 8})
    plt.show()

    plt.title(transition_file[1:-4] + f" Binary Classification Probability. Accuracy: " + str(acc[1]))
    plt.plot(np.linspace(0, lps / 14, lps), prob_sw, lw=0.5, color='purple', alpha=0.1, label='raw probabilities')
    plt.plot(np.linspace(0, lps / 14, lps),
             np.concatenate((np.zeros(int((lps - lsp) / 2)), smooth_prob, np.zeros(int((lps - lsp) / 2))), axis=None),
             lw=0.5, label='smoothed probabilities')
    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Classifying Point as Swimming")
    plt.axhline(0.5, 0, int(len(smooth_prob) / 14), lw=1, ls='-.', c='k', label='Probability = 0.5')
    plt.fill_between(np.linspace(0, lps / 14, lps), -0.2, 1.2,
                     where=(np.linspace(0, lps / 14, lps) <= sw_st + std_times[file_index][1]) & (
                                 np.linspace(0, lps / 14, lps) >= ent_water - std_times[file_index][0]), color='g',
                     alpha=0.3)
    plt.axvline(ent_water, 0, 1, lw=1, ls='-.', c='g', label='enters water')
    plt.axvline(sw_st, 0, 1, lw=1, ls='-.', c='g', label='swimming starts')
    plt.axvline(cr_st, 0, 1, lw=1, ls='-.', c='r', label='swimming stops')
    plt.axvline(dry_time, 0, 1, lw=1, ls='-.', c='r', label='bubble dries')
    plt.fill_between(np.linspace(0, lps / 14, lps), -0.2, 1.2,
                     where=(np.linspace(0, lps / 14, lps) <= dry_time + std_times[file_index][3]) & (
                                 np.linspace(0, lps / 14, lps) >= cr_st - std_times[file_index][2]), color='r',
                     alpha=0.3)
    plt.ylim([-0.2, 1.2])

    # Put a legend to the right of the current axis
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), prop={'size': 8})
    plt.xlim([0, sw_st * 3])
    plt.show()

    plt.title(transition_file[1:-4] + f" Binary Classification Probability. Accuracy: " + str(acc[1]))
    plt.plot(np.linspace(0, lps / 14, lps), prob_sw, lw=0.5, color='purple', alpha=0.1, label='raw probabilities')
    plt.plot(np.linspace(0, lps / 14, lps),
             np.concatenate((np.zeros(int((lps - lsp) / 2)), smooth_prob, np.zeros(int((lps - lsp) / 2))), axis=None),
             lw=0.5, label='smoothed probabilities')
    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Classifying Point as Swimming")
    plt.axhline(0.5, 0, int(len(smooth_prob) / 14), lw=1, ls='-.', c='k', label='Probability = 0.5')
    plt.fill_between(np.linspace(0, lps / 14, lps), -0.2, 1.2,
                     where=(np.linspace(0, lps / 14, lps) <= sw_st + std_times[file_index][1]) & (
                                 np.linspace(0, lps / 14, lps) >= ent_water - std_times[file_index][0]), color='g',
                     alpha=0.3)
    plt.axvline(ent_water, 0, 1, lw=1, ls='-.', c='g', label='enters water')
    plt.axvline(sw_st, 0, 1, lw=1, ls='-.', c='g', label='swimming starts')
    plt.axvline(cr_st, 0, 1, lw=1, ls='-.', c='r', label='swimming stops')
    plt.axvline(dry_time, 0, 1, lw=1, ls='-.', c='r', label='bubble dries')
    plt.fill_between(np.linspace(0, lps / 14, lps), -0.2, 1.2,
                     where=(np.linspace(0, lps / 14, lps) <= dry_time + std_times[file_index][3]) & (
                                 np.linspace(0, lps / 14, lps) >= cr_st - std_times[file_index][2]), color='r',
                     alpha=0.3)
    plt.ylim([-0.2, 1.2])

    # Put a legend to the right of the current axis
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), prop={'size': 8})
    plt.xlim([cr_st * 0.8, dry_time * 1.2])
    plt.show()
