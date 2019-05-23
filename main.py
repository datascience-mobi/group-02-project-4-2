# Import libraries
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scanpy as sc
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
    global centroids_array, pbmcs, genes
    pbmcs = pca_data.shape[0]
    genes = pca_data.shape[1]
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


def assign_centroids():
    global nearest_centroid
    # Assign closest Centroid
    # Loop über alle Punkte
    i = 0
    nearest_centroid = np.zeros([pbmcs, 1])
    while i < pbmcs:
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
            assign_centroids()
            empty_check()
        i += 1


def dist(cell_point, cluster_number):
    a = pca_data[cell_point, :]
    b = centroids_array[cluster_number - 1, :]
    d = np.linalg.norm(a - b)
    return d


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
    assign_centroids()

    if start == "randnum":
        empty_check()

    if t == None:

        while i < n_iterations:
            new_centroids()
            assign_centroids()
            i += 1
        improv()

    else:
        count = 0
        d = t

        while d >= t:
            new_centroids()
            assign_centroids()
            d = np.linalg.norm(centroids_oldarray-centroids_array)
            count+=1
        print("%s iterations were performed" %count)
        

    print("\nkmeans:")
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
    global dim, pca_data
    dim = d
    pca = PCA(n_components=dim)
    pca_data = pca.fit_transform(filtered_data)
    if rmo == True:
        remove_outliers()
    print("Sum of explained variances: ""%.2f" % (sum(pca.explained_variance_ratio_)) + "\n")
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
    

def sklearn_kmeans_function():
    global y_sklearnkmeans, sklearn_kmeans
    runtime_start()
    sklearn_kmeans = KMeans(n_clusters=k).fit(pca_data)
    y_sklearnkmeans = sklearn_kmeans.predict(pca_data)
    print("\nsklearn kmeans:")
    print(runtime_end())
    print("\twss: " + str(wss('sklearn')))


def plots():
    # 2D plots:
    
    # Kmeans
    fig1 = plt.figure(1, figsize=[10, 5], dpi=200)
    plt1, plt2 = fig1.subplots(1, 2)
    plt1.scatter(pca_data[:, 0], pca_data[:, 1], c=nearest_centroid_squeeze, s=5, cmap='gist_rainbow')
    plt1.plot(centroids_array[:, 0], centroids_array[:, 1], markersize=5, marker="s", linestyle='None', c='w')
    plt1.set_title('kmeans')
    
    # Sklearnkmeans
    plt2.scatter(pca_data[:, 0], pca_data[:, 1], c=y_sklearnkmeans, s=5, cmap='gist_rainbow')
    plt2.plot(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], markersize=5, marker="s", linestyle='None', c='w')
    plt2.set_title('sklearn kmeans')
    
    # 3D plots
    if dim == 3:
        fig2 = plt.figure(figsize=[15,10], dpi=200)

        # Kmeans
        plt21 = fig2.add_subplot(221, projection = '3d')
        plt21.scatter(pca_data[:, 1], pca_data[:, 2], pca_data[:, 0], s=5, c = nearest_centroid_squeeze, cmap='gist_rainbow')
        plt21.plot(centroids_array[:, 0], centroids_array[:, 1], centroids_array[:, 2], markersize=5, marker="s", linestyle='None', c='w')
        plt21.set_title('3d kmeans')

        # Sklearnkmeans
        plt22 = fig2.add_subplot(222, projection = '3d')
        plt22.scatter(pca_data[:, 1], pca_data[:, 2], pca_data[:, 0], s=5, c = y_sklearnkmeans, cmap='gist_rainbow')
        plt22.plot(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], sklearn_kmeans.cluster_centers_[:, 2], markersize=5, marker="s", linestyle='None', c='w')
        plt22.set_title('3D kmeans by sklearn')


# General Code
# Import data
data = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)

# Filter useless data & Processing
sc.pp.filter_genes(data, min_cells=1)
sc.pp.normalize_total(data)
sc.pp.log1p(data)
filtered_data = np.array(data._X.todense())

# Execute
# Startpoint selection [randnum oder randpat], Clusters, Iterations (egal wenn t), Threshhold [float oder None]
# Console dialog LEAVE COMMENTED UNTIL THE VERY END
# print("Initial cluster generation method [randnum/randcell]?")
# stringa = str(input())
# print("k?")
# inta = int(input())
# print("Maximum iterations?")
# intb = int(input())
# print("Threshold for cluster movement?")
# floata = float(input())


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #rotating 3d plot (close the other 3d plot for it to run better)
# ax.scatter(pca_data[:, 1], pca_data[:, 2], pca_data[:, 0], c = nearest_centroid_squeeze, cmap='gist_rainbow')

# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.1)

# plt.show()

pca(5, rmo=True)
kmeans('randnum', 3, 10, 0.00001)
sklearn_kmeans_function()
plots()
