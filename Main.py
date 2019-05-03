# Import libraries
import numpy as np
import pandas
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
import scanpy as sc




#define distance function which takes integer inputs which identify patient and centroid
def dist(patient_point, cluster_number):
    a = filtered_data[patient_point, :]
    b = centroids_array[cluster_number, :]
    dist = np.linalg.norm(a-b)
    return dist


# Import data
data = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)

#Filter useless data
sc.pp.filter_genes(data, min_cells=1)
filtered_data = np.array(data._X.todense())

#Create Centroid Array by randomly picking 5 patients from data  
clusters_amount = 5
centroids_numbers = np.random.randint(2700, size=clusters_amount)
centroids_array = np.empty([0, 16634])
i = 0


while i < clusters_amount:
    randompatient = centroids_numbers[i]
    centroids_array = np.append(centroids_array, [filtered_data[randompatient, :]], axis = 0)
    i += 1
