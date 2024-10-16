# Imports
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.vizualization import heatmap
import scipy.stats as ss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
np.random.seed(1)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA
import umap
import seaborn as sns

def delay_embed(angles, K):
    x_bar = np.zeros((len(angles) - K + 1, 1))
    for i in range(K-1, -1, -1):
        stop = len(angles) - K + i + 1
        x_bar = np.hstack((x_bar, angles[i:stop, :]))
    x_bar = x_bar[:, 1:]
    return x_bar

def draw_scree(pca_obj, pdf_to_draw):
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_title(f"Scree Plot for K = {i}")
    ax.plot(np.cumsum(np.hstack(([0], pca_obj.explained_variance_ratio_))))
    setpoints = np.arange(0, (i*3)//2, i//5)
    ax.vlines(setpoints, 0, 1, 'red', 'dotted', lw=0.2)
    ax.set(xlim=[0, i*10+1], xticks = np.arange(i*10+1, step=i*2), xlabel = "Num Components", ylabel = "Cumulative Variance Explained")
    pdf_to_draw.savefig(fig)

def draw_pcs(angles, pca_obj, pdf_to_draw):
    angle = angles[:, 1:]
    pcs = pca_obj.transform(StandardScaler().fit_transform(delay_embed(angle, i)))
    l = len(pcs)
    time = angles[:l, 0]
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_title(f"PCs over time for K = {i}")
    lines = ax.plot(time, pcs[:, :10], lw=0.002)
    ax.set(xlim=[min(time), max(time)], xlabel="Time")
    ax.legend(lines, ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'))
    pdf_to_draw.savefig(fig)

def draw_overall_scatter(umap_emb, behaviors, threeD=False):
    crawling, transition_1, swimming, transition_2 = umap_emb[np.where(behaviors==0)], umap_emb[np.where(behaviors==1)], umap_emb[np.where(behaviors==2)], umap_emb[np.where(behaviors==3)]
    dimensions = np.shape(umap_emb)[1]
    filename = reports_dir + f"\\{dimensions}D_UMAP_withk{i}"

    fig = plt.figure(figsize=(24,18))
    if threeD:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(crawling[:, 0], crawling[:, 1], crawling[:, 2], s=0.1, linewidths=1, c='cyan', label='crawling')
        ax.scatter(swimming[:, 0], swimming[:, 1], swimming[:, 2], s=0.1, linewidths=1, c='red', label='swimming')
        ax.scatter(transition_1[:, 0], transition_1[:, 1], transition_1[:, 2], s=3, c='green', label='transition#1')
        ax.scatter(transition_2[:, 0], transition_2[:, 1], transition_2[:, 2], s=3, c='magenta', label='transition#2')
        ax.set_title(f"3D Scatter Plot for K = {i}")
        filename = filename + "_3D"
    else:
        ax = fig.add_subplot(111)
        ax.scatter(crawling[:, 0], crawling[:, 1], s=0.1, linewidths=1, c='cyan', label='crawling')
        ax.scatter(swimming[:, 0], swimming[:, 1], s=0.1, linewidths=1, c='red', label='swimming')
        ax.scatter(transition_1[:, 0], transition_1[:, 1], s=3, c='green', label='transition#1')
        ax.scatter(transition_2[:, 0], transition_2[:, 1], s=3, c='magenta', label='transition#2')
        ax.set_title(f"2D Scatter Plot for K = {i}")
        filename = filename + "_2D"
    plt.legend()
    plt.savefig(filename, dpi=600)

def draw_file_scatter(umap_emb, behaviors, file, threeD=False, behaviors_pred_curr=None):
    start, end = data_pieces[file]
    umap_curr, behavior_curr = umap_emb[start:end, :], behaviors[start:end]
    crawling, transition_1, swimming, transition_2 = umap_curr[behavior_curr==0], umap_curr[behavior_curr==1], umap_curr[behavior_curr==2], umap_curr[behavior_curr==3]
    dimensions = np.shape(umap_curr)[1]
    if behaviors_pred_curr is None:
        filename = reports_dir + f"\\{dimensions}D_UMAP_withk{i}_for_{file[42:-4]}"
        fig = plt.figure(figsize=(24,18))
        if threeD:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(crawling[:, 0], crawling[:, 1], crawling[:, 2], s=3, c='blue', label='crawling')
            ax.scatter(swimming[:, 0], swimming[:, 1], swimming[:, 2], s=3, c='maroon', label='swimming')
            ax.scatter(transition_1[:, 0], transition_1[:, 1], transition_1[:, 2], s=5, c='limegreen', label='transition#1')
            ax.scatter(transition_2[:, 0], transition_2[:, 1], transition_2[:, 2], s=5, c='magenta', label='transition#2')
            ax.set_title(f"3D Scatter Plot for K = {i}, file = {file[42:]}")
            filename = filename + "_3D"
        else:
            ax = fig.add_subplot(111)
            ax.scatter(crawling[:, 0], crawling[:, 1], s=3, c='blue', label='crawling')
            ax.scatter(swimming[:, 0], swimming[:, 1], s=3, c='maroon', label='swimming')
            ax.scatter(transition_1[:, 0], transition_1[:, 1], s=5, c='limegreen', label='transition#1')
            ax.scatter(transition_2[:, 0], transition_2[:, 1], s=5, c='magenta', label='transition#2')
            ax.set_title(f"2D Scatter Plot for K = {i}, file = {file[42:]}")
            filename = filename + "_2D"
    else:
        inds = np.logical_or(np.logical_or(np.logical_or(np.logical_and(behavior_curr==0, behaviors_pred_curr!=0), np.logical_and(behavior_curr==1, behaviors_pred_curr!=1)), np.logical_and(behavior_curr==2, behaviors_pred_curr!=2)), np.logical_and(behavior_curr==3, behaviors_pred_curr!=3))
        errors = umap_curr[inds]
        filename = reports_dir + f"\\{dimensions}D_UMAP_withk{i}_for_{file[42:-4]}"
        fig = plt.figure(figsize=(36,18))
        if threeD:
            ax = fig.add_subplot(221, projection='3d')
            ax.scatter(crawling[:, 0], crawling[:, 1], crawling[:, 2], s=1, c='blue', label='crawling')
            ax.scatter(swimming[:, 0], swimming[:, 1], swimming[:, 2], s=1, c='maroon', label='swimming')
            ax.scatter(transition_1[:, 0], transition_1[:, 1], transition_1[:, 2], s=2, c='limegreen', label='transition#1')
            ax.scatter(transition_2[:, 0], transition_2[:, 1], transition_2[:, 2], s=2, c='magenta', label='transition#2')
            ax.set_title(f"True 3D Scatter Plot for K = {i}, file = {file[42:]}")
            ax = fig.add_subplot(222, projection='3d')
            ax.scatter(crawling[:, 0], crawling[:, 1], crawling[:, 2], s=1, c='blue', label='crawling')
            ax.scatter(swimming[:, 0], swimming[:, 1], swimming[:, 2], s=1, c='maroon', label='swimming')
            ax.scatter(transition_1[:, 0], transition_1[:, 1], transition_1[:, 2], s=2, c='limegreen', label='transition#1')
            ax.scatter(transition_2[:, 0], transition_2[:, 1], transition_2[:, 2], s=2, c='magenta', label='transition#2')
            ax.scatter(errors[:, 0], errors[:, 1], errors[:, 2], s=4, c='orange', label='errors')
            ax.set_title(f"Predicted 3D Scatter Plot for K = {i}, file = {file[42:]}", fontsize=20)
            plt.legend(fontsize='20', markerscale=5)
            ax = fig.add_subplot(212)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            cm = confusion_matrix(behavior_curr, behaviors_pred_curr)
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
            ax.set_title(f'Confusion Matrix of {file[42:-4]}, accuracy = {acc}', fontsize=18)
            ax.set_xlabel('True Label', fontsize=20)
            ax.set_ylabel('Predicted Label', fontsize=20)
            ax.set_xticklabels(['crawling', 'transition1', 'swimming', 'transition2'])
            ax.set_yticklabels(['crawling', 'transition1', 'swimming', 'transition2'])
            filename = filename + "_3D"
        else:
            ax = fig.add_subplot(221)
            ax.scatter(crawling[:, 0], crawling[:, 1], s=1, c='blue', label='crawling')
            ax.scatter(swimming[:, 0], swimming[:, 1], s=1, c='maroon', label='swimming')
            ax.scatter(transition_1[:, 0], transition_1[:, 1], s=2, c='limegreen', label='transition#1')
            ax.scatter(transition_2[:, 0], transition_2[:, 1], s=2, c='magenta', label='transition#2')
            ax.set_title(f"True 2D Scatter Plot for K = {i}, file = {file[42:]}", fontsize=20)
            ax = fig.add_subplot(222)
            ax.scatter(crawling[:, 0], crawling[:, 1], s=1, c='blue', label='crawling')
            ax.scatter(swimming[:, 0], swimming[:, 1], s=1, c='maroon', label='swimming')
            ax.scatter(transition_1[:, 0], transition_1[:, 1], s=2, c='limegreen', label='transition#1')
            ax.scatter(transition_2[:, 0], transition_2[:, 1], s=2, c='magenta', label='transition#2')
            ax.scatter(errors[:, 0], errors[:, 1], s=4, c='orange', label='errors')
            ax.set_title(f"Predicted 2D Scatter Plot for K = {i}, file = {file[42:]}", fontsize=20)
            plt.legend(fontsize='20', markerscale=5)
            ax = fig.add_subplot(212)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            cm = confusion_matrix(behavior_curr, behaviors_pred_curr)
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
            ax.set_title(f'Confusion Matrix of {file[42:-4]}, accuracy = {acc}', fontsize=18)
            ax.set_xlabel('True Label', fontsize=20)
            ax.set_ylabel('Predicted Label', fontsize=20)
            ax.set_xticklabels(['crawling', 'transition1', 'swimming', 'transition2'])
            ax.set_yticklabels(['crawling', 'transition1', 'swimming', 'transition2'])
            filename = filename + "_2D"
    plt.savefig(filename+".png", dpi=500)
    plt.close()
    
def draw_kymoethogram(x, probs, file):
    start, end = data_pieces[file]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(36,24), sharex=True, layout='constrained')
    time = times[file]
    time = time[:-14]
    angles = np.loadtxt(file, dtype='float', skiprows=1)
    angles = angles[:, 1:]
    heatmap(time, angles, clip=None, timestep=15, fig=fig, ax=ax1, title=f"Kymograph for {file}")
    xtimes = np.arange(min(time), max(time), 15)
    xticks = []
    for i in xtimes:
        ind = np.searchsorted(time, i)
        xticks.append(ind)
        xticklabels = [str(i) for i in xtimes]
    time = time*14
    ax2.set(xlim=[np.min(time), np.max(time)])
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    ax2.plot(time, probs[:, 0], label='crawling prob', lw=0.5)
    ax2.plot(time, probs[:, 1], label='transition1 prob', lw=0.5)
    ax2.plot(time, probs[:, 2], label='swimming prob', lw=0.5)
    ax2.plot(time, probs[:, 3], label='transition2 prob', lw=0.5)
    ax2.set_title(f"Ethogram for {file}", fontsize=20)
    ax2.set_xlabel('Time', fontsize=20)
    ax2.set_ylabel('Prediction Probabilities', fontsize=20)
    plt.legend(fontsize=20, markerscale=8)
    filename = reports_dir+'\\'+file[42:-4]+'probs_eth.png'
    plt.savefig(filename, dpi=500)
    plt.close()

if __name__ == "__main__":
    i=15
    path = r"C:\Users\arnav\Documents\data\Transitions"
    timestamp_path = r"C:\Users\arnav\Documents\wopodyn"
    # Import timestamps of human-observed behavior transitions
    timestamps_keys = np.loadtxt(timestamp_path+r'\transition-times.txt', dtype='str', delimiter=',', usecols=0)
    timestamps_values = np.loadtxt(timestamp_path+r'\transition-times.txt', delimiter=',', dtype='float', usecols=range(1, 5))
    timestamps = {path+timestamps_keys[k]: timestamps_values[k] for k in range(len(timestamps_keys))}  

    # Import Data
    x, y, filenames, data_pieces, times, k = np.zeros((1, 10*i)), np.zeros((1,)), [], {}, {}, 0        
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
            times[curr_file]=to_add[:, 0]
            x_curr = delay_embed(StandardScaler().fit_transform(to_add[:, 1:]), i)
            x = np.vstack((x, x_curr))                                                                                                          #delay embed file data and add it to overall X dataset
            y = np.hstack((y, np.array([ss.mode(y_temp[c:c+i-1], axis=None)[0] for c in range(len(y_temp)-i+1)])))                                              #function to transform behavior labels onto delay embedded dataset
            data_pieces[curr_file] = (k, k+len(x_curr))
            k = k+len(x_curr)
    x, y = StandardScaler().fit_transform(x[1:, :]), y[1:]                                                                                                                                #skip out first row of dataset (set to row of zeros during instantiation)
    # Define the folder and filename where the .pdf file should be saved
    reports_dir = r"C:\Users\arnav\Documents\wopodyn-main\reports\trials"

    # filename = os.path.join(reports_dir, f"umap-scatters-{i}.pdf")
    # pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    pca = PCA(n_components=10*i)                                                                                                                #init pca object
    pcs_overall = pca.fit_transform(x)                                                                                                          #fit and transform the angles data
    elbow_i = i*3//5
    pcs_umap = pcs_overall[:, :elbow_i]

    twodreduction = umap.UMAP(a=None, angular_rp_forest=False, b=None, force_approximation_algorithm=False, init='spectral', learning_rate=1.0, local_connectivity=1.0, low_memory=False, metric='euclidean', metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None, n_neighbors=15, negative_sample_rate=5, output_metric='euclidean', output_metric_kwds=None, repulsion_strength=1.0, set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical', target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5, transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)         
    threedreduction = umap.UMAP(a=None, angular_rp_forest=False, b=None, force_approximation_algorithm=False, init='spectral', learning_rate=1.0, local_connectivity=1.0, low_memory=False, metric='euclidean', metric_kwds=None, min_dist=0.1, n_components=3, n_epochs=None, n_neighbors=15, negative_sample_rate=5, output_metric='euclidean', output_metric_kwds=None, repulsion_strength=1.0, set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical', target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5, transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)         
    # fourdreduction = umap.UMAP(a=None, angular_rp_forest=False, b=None, force_approximation_algorithm=False, init='spectral', learning_rate=1.0, local_connectivity=1.0, low_memory=False, metric='euclidean', metric_kwds=None, min_dist=0.1, n_components=4, n_epochs=None, n_neighbors=15, negative_sample_rate=5, output_metric='euclidean', output_metric_kwds=None, repulsion_strength=1.0, set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical', target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5, transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)         
    twod_embedding = twodreduction.fit_transform(StandardScaler().fit_transform(pcs_umap))
    threed_embedding = threedreduction.fit_transform(StandardScaler().fit_transform(pcs_umap))
    # fourd_embedding = fourdreduction.fit_transform(StandardScaler().fit_transform(pcs_umap))
    
    

    accs = []
    for file in filenames:
        # grid_params = { 'n_neighbors' : [150, 175, 200, 225, 250], 'weights' : ['distance'],
        #         'metric' : ['manhattan']}
        # gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=2, n_jobs = -1)
        start, end = data_pieces[file]
        x_train, x_test, y_train, y_test = np.vstack((x[:start], x[end:])), x[start:end], np.concatenate((y[:start], y[end:])), y[start:end]
        # g_fit = gs.fit(x_train, y_train)
        # print(f"Best accuracy of {g_fit.best_score_} with parameters: {g_fit.best_params_}")
        # clsfr.fit(x_train, y_train)
        # y_hat = clsfr.predict(x_test)
        # print(f"Overall accuracy for {clsfr}: {accuracy_score(y_test, y_hat)}")    
        nn = 100+((end-start)//60)
        clsfr = KNeighborsClassifier(metric='manhattan', n_neighbors=nn, weights='distance')
        clsfr.fit(x_train, y_train)
        y_hat = clsfr.predict(x_test)
        probs = clsfr.predict_proba(x_test)
        acc = accuracy_score(y_test, y_hat)
        accs.append(acc)
        print(acc)
        draw_file_scatter(twod_embedding, y, file, False, y_hat)
        # draw_file_scatter(threed_embedding, y, file, False, y_hat)
        draw_file_scatter(threed_embedding, y, file, True, y_hat)
        # draw_file_scatter(fourd_embedding, y, file, False, y_hat)
        # draw_file_scatter(fourd_embedding, y, file, True, y_hat)
        draw_kymoethogram(x, probs, file)

    # for file in filenames:
    #     draw_file_scatter(twod_embedding, y, file)
    # for file in filenames:
    #     draw_file_scatter(threed_embedding, y, file)
    # for file in filenames:
    #     draw_file_scatter(threed_embedding, y, file, True)
    # for file in filenames:
    #     draw_file_scatter(fourd_embedding, y, file)
    # for file in filenames:
    #     draw_file_scatter(fourd_embedding, y, file, True)

    # draw_overall_scatter(twod_embedding, y)
    # draw_overall_scatter(threed_embedding, y)
    # draw_overall_scatter(threed_embedding, y, True)
    # draw_overall_scatter(fourd_embedding, y)
    # draw_overall_scatter(fourd_embedding, y, True)

    # draw_scree(pca, pdf)
    # for file in filenames:
    #     data = np.loadtxt(file, dtype='float', skiprows=1)
    #     draw_pcs(data, pca, pdf)
    # pdf.close()
