# Import libraries
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scanpy as sc
from random import randint
from matplotlib import colors
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from mpl_toolkits.mplot3d import Axes3D


# Functions
# Define distance function which takes integer inputs which identify cell and centroid
def runtime_start():
    global t1
    t1 = datetime.now().time()


def runtime_end():
    t2 = datetime.now().time()
    fmt = '%H:%M:%S.%f'
    elapsed = str(datetime.strptime(str(t2), fmt) - datetime.strptime(str(t1), fmt))
    return str("\truntime: " + elapsed)


def random_start_centroids(starttype):
    # Create Centroid Array by randomly picking k pbmcs from data
    global centroids_array
    
    centroids_array = np.empty([0, genes])

    if starttype == "randcell":
        centroids_numbers = np.random.randint(pbmcs, size=k)
        i = 0
        # Pick random start sample 
        while i < k:
            random_cell = centroids_numbers[i]
            centroids_array = np.append(centroids_array, [pca_data[random_cell, :]], axis=0)
            i += 1

    elif starttype == "randnum":
        centroids_array = (np.amax(pca_data) - np.amin(pca_data)) * np.random.random_sample((k, genes)) + np.amin(
            pca_data)
    
    elif starttype == "k++":
        kppcentroids()
    
    return centroids_array

def dist(cell_point, cluster_number):

    return np.linalg.norm(pca_data[cell_point, :] - centroids_array[cluster_number - 1, :])


def assign_centroids(data_array):
    global nearest_centroid
    # Assign closest Centroid
    i = 0
    array_dim1 = data_array.shape[0]
    nearest_centroid = np.zeros([array_dim1, 1])
    
    # Loop über alle Punkte
    while i < array_dim1:
        sml_distance = 0

        # While loop selecting every centroid
        j = 1
        while j <= k:

            if sml_distance == 0 or dist(i, j) < sml_distance:
                sml_distance = dist(i, j)
                nearest_centroid[i, 0] = j
            j += 1
        i += 1


def empty_check():
    # Sicherstellen dass es durch die Zufallscentroids keine leeren Cluster gibt
    i = 0
    while i < k:
        if list(nearest_centroid).count(i + 1) == 0:
            print("Empty cluster! Correcting centroids.")
            random_start_centroids("randnum")
            assign_centroids(pca_data)
            empty_check()
        i += 1


def new_centroids():
    global centroids_array, centroids_oldarray, nearest_centroid_squeeze
    centroids_oldarray = centroids_array # create copy of old array for threshold funcion
    nearest_centroid_squeeze = np.squeeze(nearest_centroid.astype(int))
    centroids_array = np.empty([0, genes])

    i = 1
    while i <= k:
        calc_means = np.mean(pca_data[nearest_centroid_squeeze == i], axis = 0)
        centroids_array = np.append(centroids_array, np.expand_dims(calc_means, axis = 0), axis = 0)
        i += 1
    

def kppcentroids():
    global centroids_array, dist_array, prob_array
    first_centroid = np.random.randint(pbmcs, size=1)
    i = 0
    centroids_array = np.append(centroids_array, pca_data[first_centroid, :], axis=0)
    dist_array = np.empty ([0,pbmcs])
    prob_array = np.empty ([0,pbmcs])
    j = 0
     
    while i < k - 1:
        z = centroids_array.shape[0] + 1 #+1 weil die dist function clusternumber -1 macht
        while j < pbmcs:
            sml_distance = -1
            l = 1 # wieder weil die cluster number -1 ist
            while l < z:
                if sml_distance == -1 or dist(j, l) < sml_distance:
                    sml_distance = dist(j, l)
                l += 1
            dist_array = np.append(dist_array,sml_distance **2)        
            j += 1
        #centroids_array um neuen centroid erweitern
        prob_array = dist_array / np.sum(dist_array)
        s = np.random.choice(pbmcs,p = prob_array )
        centroids_array = np.append(centroids_array, np.expand_dims(pca_data[s, :], axis=0), axis=0)
        i += 1
    return (centroids_array)

# Function giving distance between clusters after n iterations            
def improv():
    distances = []
    i = 0
    while i < k: 
        d = np.linalg.norm(centroids_array[i, :] - centroids_oldarray[i,:])
        distances.append(d)
        i += 1
    c_str = np.array2string(np.array(distances), precision=2)
    print("Distances of clusters as compared to last generation: \n" + str(c_str))
  

def kmeans(start, k1, n_iterations, t):
    global k
    k = k1
    i = 0
    runtime_start()

    random_start_centroids(start)
    assign_centroids(pca_data)

    if start == "randnum":
        empty_check()

    if t == None:

        while i < n_iterations:
            new_centroids()
            assign_centroids(pca_data)
            i += 1
        improv()

    else:
        count = 0
        d = t

        while d >= t:
            new_centroids()
            assign_centroids(pca_data)
            d = np.linalg.norm(centroids_oldarray-centroids_array)
            count+=1
        print("%s iterations were performed" %count)
        
    print("\nKMEANS:")
    print("\ngroup 4_2 algorithm:")
    print(runtime_end())
    print("\twss: " + str(wss('self')))

def minibatch(k1, n_iterations, b):
    global k, pca_data, nearest_centroid_squeeze, pca_data, bg, n_iterationsg, centroids_array, cnnew
    k = k1
    bg = b
    n_iterationsg = n_iterations
    runtime_start()
    v = np.zeros((k, 1))
    j = 1
    random_start_centroids("randcell")
    cnnew = centroids_array
    while (j <= n_iterations):
        # Reduce data to batch
        pca_batch = pca_data[np.random.randint(pca_data.shape[0], size=b), :]
        # Start centroids
        assign_centroids(pca_batch)
        i = 0
        while (i < b):
            c = cnnew[int(nearest_centroid[i, 0])-1, :]
            v[int((nearest_centroid[i, 0]-1)), 0] =  int(v[int((nearest_centroid[i, 0]-1)), 0]) + 1
            n = 1/v[int((nearest_centroid[i, 0]-1)), 0]
            cnnew[int(nearest_centroid[i, 0])-1, :] = c * (1-n) + pca_data[i, :] * n
            i+=1
        j+=1

    centroids_array = cnnew
    assign_centroids(pca_data)
    nearest_centroid_squeeze = np.squeeze(nearest_centroid.astype(int))
    print("\nMINI-BATCH:")
    print("\ngroup 4_2 algorithm:")
    print(runtime_end())
    print("\twss: " + str(wss('self')))
        


# calculates sum of the squared distance in each cluster
def wss(where):
        i = 0
        wsssum = 0
        while (i < len(pca_data)):
            if where == "self":
                assigned_centroid = int(nearest_centroid[i,0])
                centr_val = centroids_array[assigned_centroid-1]
            if where == "sklearn":
                assigned_centroid = int(y_sklearnkmeans[i])
                centr_val = sklearn_kmeans.cluster_centers_[assigned_centroid]
            point_val = pca_data[i] 
            i+=1
            sqdist = np.linalg.norm(centr_val - point_val)**2
            wsssum += np.trunc(sqdist)              
        return(wsssum)


def remove_outliers():
    global pca_data
    X_train = pca_data
    clf = IsolationForest(behaviour="new", contamination=.07, max_samples=0.25)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    pca_data = X_train[np.where(y_pred_train == 1, True, False)]

# PCA
def pca(d, rmo=False):
    global dim, pca_data, pbmcs, genes

    dim = d
    pca = PCA(n_components=dim)
    pca_data = pca.fit_transform(filtered_data)
    if rmo == True:
        remove_outliers()
    print("Sum of explained variances: ""%.2f" % (sum(pca.explained_variance_ratio_)) + "\n")
    
    pbmcs = pca_data.shape[0]
    genes = pca_data.shape[1]
    # print(pca.singular_values_)


#Ellbow PCA
def ellbow_pca(components):
    test_array = np.empty([0])
    pca = PCA(n_components=components)
    pca.fit_transform(filtered_data)
    n = 0
    while(n <= components):
        variance = sum(pca.explained_variance_ratio_[0:n])
        test_array = np.append(test_array, variance)
        n+=1
    plt.plot(test_array)
    plt.xlabel('n PCA')
    plt.ylabel('explained variance')
    plt.show()


def ellbow_cluster(where = "self", clusters = 4, var = "kmeans"):
    k = 1
    sq_dist = 0
    sq_dist_array = np.empty([0])
    while (k<= clusters):
        if var == "mini":
            minibatch(k, 10, 2000)
            sq_dist = wss("self")
            
        if where == "self":
            kmeans("k++", k, 20, 0.00001)
            sq_dist = wss("self")
        if where == "sklearn":
           sklearn_kmeans_function(var,k,"k++")
           sq_dist = wss("sklearn") 
        sq_dist_array = np.append(sq_dist_array,sq_dist)
        
        k+=1
    print(sq_dist_array)
    plt.plot(np.arange(1,clusters+1), sq_dist_array)
    plt.show()    


def sklearn_kmeans_function(var, k, start):
    global y_sklearnkmeans, sklearn_kmeans, pca_data
    runtime_start()
    if start == "randcell" or start == "randnum": 
        if var == "kmeans":
            sklearn_kmeans = KMeans(init='random', n_clusters=k).fit(pca_data)
        if var == "mini":
            sklearn_kmeans = MiniBatchKMeans(n_clusters=k, init = 'random', max_iter=n_iterationsg, batch_size=bg).fit(pca_data)
    if start == "k++": 
        if var == "kmeans":
            sklearn_kmeans = KMeans(n_clusters=k).fit(pca_data)
        if var == "mini":
            sklearn_kmeans = MiniBatchKMeans(n_clusters=k, max_iter=n_iterationsg, batch_size=bg).fit(pca_data)
    y_sklearnkmeans = sklearn_kmeans.predict(pca_data)
    print("\nsklearn kmeans (best of 10):")
    print(runtime_end())
    print("\twss: " + str(wss('sklearn')))


def plots(add = ""):
    global fig1, fig2
    # 2D plots:
    additional = ""
    if add == "mini":
        additional = " (mini-batch)"
    # Kmeans
    fig1 = plt.figure(1, figsize=[10, 5], dpi=200)
    plt1, plt2 = fig1.subplots(1, 2)
    plt1.scatter(pca_data[:, 0], pca_data[:, 1], c=nearest_centroid_squeeze, s=0.5, cmap='gist_rainbow')
    plt1.plot(centroids_array[:, 0], centroids_array[:, 1], markersize=5, marker="s", linestyle='None', c='w')
    plt1.set_title('kmeans' + additional)
    
    # Sklearnkmeans
    plt2.scatter(pca_data[:, 0], pca_data[:, 1], c=y_sklearnkmeans, s=0.5, cmap='gist_rainbow')
    plt2.plot(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], markersize=5, marker="s", linestyle='None', c='w')
    plt2.set_title('sklearn kmeans' + additional)
    
    # 3D plots
    if dim >= 3:
        fig2 = plt.figure(2, figsize=[15,10], dpi=200)

        # Kmeans
        plt21 = fig2.add_subplot(221, projection = '3d')
        plt21.scatter(pca_data[:, 1], pca_data[:, 2], pca_data[:, 0], s=2, c = nearest_centroid_squeeze, cmap='gist_rainbow')
        plt21.plot(centroids_array[:, 0], centroids_array[:, 1], centroids_array[:, 2], markersize=5, marker="s", linestyle='None', c='w')
        plt21.set_title('3d kmeans' + additional)

        # Sklearnkmeans
        plt22 = fig2.add_subplot(222, projection = '3d')
        plt22.scatter(pca_data[:, 1], pca_data[:, 2], pca_data[:, 0], s=2, c = y_sklearnkmeans, cmap='gist_rainbow')
        plt22.plot(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], sklearn_kmeans.cluster_centers_[:, 2], markersize=5, marker="s", linestyle='None', c='w')
        plt22.set_title('3D kmeans by sklearn' + additional)


def cluster(pcas = 5, rmo=True, variant = 'kmeans', start='randcell', k = 3, max_iterations = 10, threshold = 0.00001, batch_size = 2000, hd = False):
    pca(pcas, rmo)
    if hd == True:
        highlightdiffs(start, k , max_iterations, threshold, batch_size)
    else:
        if variant == "kmeans":
            kmeans(start, k, max_iterations, threshold)
            sklearn_kmeans_function("kmeans", k, start)
            # plots()

        if variant == "mini":
            minibatch(k, max_iterations, batch_size)
            sklearn_kmeans_function("mini", k, start)
            # plots("mini")


def highlightdiffs(start=None, k=None, max_iterations=None, threshold=None, batch_size=None, between="self"):
    global nearest_centroid_squeeze, centroids_array

    if between == "reg/mb":
        # Performing KMeans clustering and saving results
        kmeans(start, k, max_iterations, threshold)
        centroids_array = centroids_array[centroids_array[:,0].argsort()]
        assign_centroids(pca_data)
        vr = np.squeeze(nearest_centroid.astype(int))

        # Performing Mini-Batch KMeans clustering and saving results
        minibatch(k, max_iterations, batch_size)
        centroids_array = centroids_array[centroids_array[:,0].argsort()]
        assign_centroids(pca_data)
        vm = np.squeeze(nearest_centroid.astype(int))
    
    if between == "self":
        vr = nearest_centroid_squeeze
        i = 0
        l = np.size(np.unique(vr))
        while i < l:
            # Select indices of all highest cluster index elements
            a = np.where(vr == l - i)
            # Find of those the one closest to the assigned center as it is least likely to be an outlier
            b = np.subtract(pca_data[a, :], centroids_array[(l-i-1), :])
            b = np.squeeze(b[0, :, :])
            ct = np.absolute(np.sum(b, axis=1))
            # Return index with regards to a as to where the closest data point is
            ctt = np.where(ct == np.amin(ct))
            # Select the same datapoint in sklearn clustering and reassign the new number to all elements of indifferently clusterd elements
            c = np.where(y_sklearnkmeans == y_sklearnkmeans[a[0][ctt]])
            y_sklearnkmeans[c] = vr[a[0][ctt]]
            i+=1
        vm = np.squeeze(y_sklearnkmeans.astype(int)) 

        # Compare results of both algorithms and find differences
        vn = np.where(np.subtract(vr, vm) == 0)[0]
        nearest_centroid_squeeze = np.squeeze(np.zeros(np.size(nearest_centroid)).astype(int))
        np.put(nearest_centroid_squeeze, vn, 1)
        diffsabso = np.count_nonzero(nearest_centroid_squeeze == 0)
        diffsperc = round(diffsabso/np.size(nearest_centroid_squeeze)*100, 3)
        print("The algorithms assigned " + str(diffsperc) + "% (" + str(diffsabso) + ") of the points differently.")

    # Plot
    fig3 = plt.figure(3, figsize=[5, 5], dpi=200)
    plt3 = fig3.subplots(1)
    if diffsabso == 0:
        plt3.scatter(pca_data[:, 0], pca_data[:, 1], c='white', s=0.5)
    else:
        plt3.scatter(pca_data[:, 0], pca_data[:, 1], c=nearest_centroid_squeeze, s=0.5, cmap=colors.ListedColormap(['red', 'white']))
    plt3.set_title('differences')
    white_patch = mpatches.Patch(color='white', label='Identical')
    red_patch = mpatches.Patch(color='red', label='Different')
    plt.legend(handles=[white_patch, red_patch])

    nearest_centroid_squeeze = vr


def multi_cluster(start = 'k++', k = 3, n_init = 10):
    global nearest_centroid_squeeze, centroids_array
    pca(5, rmo = True)
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    runtime_start()
    clusteringarray = np.empty([pca_data.shape[0], n_init])
    centroidsarray = np.empty([n_init, 3, pca_data.shape[1]])
    wssarray = []

    i = 0
    while i<n_init:
        kmeans(start, k, 30, 0.0001)
        wssarray.append(wss('self'))
        clusteringarray[:, i] = nearest_centroid_squeeze
        centroidsarray[i, :, :] = centroids_array
        i += 1

    bestwssindex = wssarray.index(min(wssarray))
    bestclustering = np.squeeze(clusteringarray[:, bestwssindex]).astype(int)

    sys.stdout.close()
    sys.stdout = original_stdout
    print("\nKMEANS:")
    print("\ngroup 4_2 algorithm (best of " + str(n_init) + "):")
    print(runtime_end())
    print("\twss: " + str(min(wssarray)))

    sklearn_kmeans_function("kmeans", k, start)
    nearest_centroid_squeeze = bestclustering
    centroids_array[:, 0] = centroidsarray[bestwssindex, :, 0]
    centroids_array[:, 1] = centroidsarray[bestwssindex, :, 1]
    plots()

    

# General Code
# Import data
data = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)

# Filter useless data & Processing
sc.pp.filter_genes(data, min_cells=1)
sc.pp.normalize_total(data)
sc.pp.log1p(data)
filtered_data = np.array(data._X.todense())

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #rotating 3d plot (close the other 3d plot for it to run better)
# ax.scatter(pca_data[:, 1], pca_data[:, 2], pca_data[:, 0], c = nearest_centroid_squeeze, cmap='gist_rainbow')

# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.1)

multi_cluster()