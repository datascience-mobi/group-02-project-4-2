# Import libraries
import numpy as np
import pandas
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
import scanpy as sc
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from mpl_toolkits.mplot3d import Axes3D

# Global Variables
t1 = 0
pbmcs = 0
genes = 0
centroids_array = 0
nearest_centroid = 0
k = 0
dim = 0
pca_data = []


# Functions
# Define distance function which takes integer inputs which identify cell and centroid
def runtime_start():
    global t1
    t1 = datetime.now().time()


def runtime_end():
    t2 = datetime.now().time()
    fmt = '%H:%M:%S.%f'
    elapsed = str(datetime.strptime(str(t2), fmt) - datetime.strptime(str(t1), fmt))
    return str("runtime: " + elapsed)


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
    global centroids_array, centroids_oldarray
    centroids_oldarray = centroids_array # create copy of old array for threshold funcion
    zeros = np.zeros([pbmcs, 1])
    centroids_array = np.empty([0, genes])
    # "Masken" um values aus pca_data abzurufen
    nearest_centroidpca1 = np.append(nearest_centroid, zeros, axis=1)
    nearest_centroidpca2 = np.append(zeros, nearest_centroid, axis=1)

    if dim ==3:
        nearest_centroidpca1 = np.append(nearest_centroidpca1, zeros, axis=1)
        nearest_centroidpca2 = np.append(nearest_centroidpca2, zeros, axis=1)
        nearest_centroidpca3 = np.append(zeros, zeros, axis=1)
        nearest_centroidpca3 = np.append(nearest_centroidpca3, nearest_centroid, axis=1)
    # while loop der für alle k cluster läuft:
    i = 1
    while i <= k:
        pca1 = np.mean(pca_data[nearest_centroidpca1 == i])
        pca2 = np.mean(pca_data[nearest_centroidpca2 == i])
        if dim == 3:
            pca3 = np.mean(pca_data[nearest_centroidpca3 == i])
            centroids_array = np.append(centroids_array, [[pca1, pca2, pca3]], axis=0)
        else:
            centroids_array = np.append(centroids_array, [[pca1, pca2]], axis=0)
        i += 1

# Clustering threshold, centroid arrays have the dimension k, genes, repeat until distance is smaller than t
def thresh(t1):
    global centroids_array, centroids_oldarray, k
    t = t1  # Threshold to determine when algorithm is done
    i = 0
    c = 1 # Add counter to determine how many cycles have passed
    while i < k: 
        a = centroids_array[i,:]
        b = centroids_oldarray[i,:]
        d = np.linalg.norm(a-b)
        if d < t:
            i += 1 
        elif d >= t:
            new_centroids()
            assign_centroids()
            c += 1
    print (str(c) + " iterations were performed")
    # Können wir wenn wir wollen dann ans ende von kmeans packen anstelle des while loops


# Function giving distance between clusters after n iterations            
def improv():      
    global centroids_array, centroids_oldarray, k
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
    random_start_centroids(start)
    assign_centroids()
    if start == "randnum":
        empty_check()
    if t == None:
        while i < n_iterations:
            new_centroids()
            assign_centroids()
            i += 1
    else:
        new_centroids()
        assign_centroids()
        thresh(t)
    improv()

# calculates sum of the squared distance in each cluster
def wss(where):
        i = 0
        wsssum = 0
        while (i < len(pca_data)):
                if where == "self":
                    assigned_centroid = int(nearest_centroid[i,0])
                if where == "sklearn":
                    assigned_centroid = int(y_sklearnkmeans[i])
                centr_val = centroids_array[assigned_centroid-1]
                point_val = pca_data[i] 
                i+=1
                sqdist = np.linalg.norm(centr_val - point_val)**2
                wsssum += sqdist              
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
    n=0
    test_array = np.empty([0])
    while(n<components):
        pca = PCA(n_components=n)
        temp = pca.fit_transform(filtered_data)
        variance = sum(pca.explained_variance_ratio_)
        n+=1
        test_array = np.append(test_array, variance)
    pyplot.plot(test_array)

# General Code
# Import data
data = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)

# Filter useless data & Processing
sc.pp.filter_genes(data, min_cells=1)
sc.pp.normalize_per_cell(data, counts_per_cell_after=1e4)
sc.pp.log1p(data)
filtered_data = np.array(data._X.todense())
pca(2, rmo=True)

# Execute
runtime_start()

# Startpoint selection [randnum oder randpat], Clusters, Iterations (egal wenn t), Threshhold [float oder None]
kmeans("randnum", 3, 10, 0.1)

print("\nkmeans:")
print(runtime_end())

# plotting
fig = pyplot.figure(1, figsize=[10, 5], dpi=200)
plt1, plt2 = fig.subplots(1, 2)
nearest_centroid_squeeze = np.squeeze(nearest_centroid.astype(int))
plt1.scatter(pca_data[:, 0], pca_data[:, 1], c=nearest_centroid_squeeze, s=20, cmap='viridis')
plt1.set_title('kmeans')
a_str = np.array2string(centroids_array[np.argsort(centroids_array[:, 0])], precision=2, separator=' ')
print("centroids: \n" + ' ' + a_str[1:-1])

# sklearn comparison
print("\nsklearn kmeans:")
runtime_start()
sklearn_kmeans = KMeans(n_clusters=k).fit(pca_data)
y_sklearnkmeans = sklearn_kmeans.predict(pca_data)
print(runtime_end())
plt2.scatter(pca_data[:, 0], pca_data[:, 1], c=y_sklearnkmeans, s=20, cmap='viridis')
plt2.set_title('sklearn kmeans')
pyplot.show()
b_str = np.array2string(sklearn_kmeans.cluster_centers_[np.argsort(sklearn_kmeans.cluster_centers_[:, 0])], precision=2, separator=' ')
print("centroids: \n" + ' ' + b_str[1:-1])

if dim == 3:
    fig2 = pyplot.figure(figsize=[10,5], dpi=200)
    plt21 = fig2.add_subplot(221, projection = '3d')
    plt21.scatter(pca_data[:, 1], pca_data[:, 2], pca_data[:, 0], c = nearest_centroid_squeeze, cmap='viridis')
    plt21.set_title('3d kmeans')
    plt22 = fig2.add_subplot(222, projection = '3d')
    plt22.scatter(pca_data[:, 1], pca_data[:, 2], pca_data[:, 0], c = y_sklearnkmeans, cmap='viridis')
    plt22.set_title('3D kmeans by sklearn')
