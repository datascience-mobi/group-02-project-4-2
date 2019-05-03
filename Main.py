# Import libraries
import numpy as np
import pandas 
import matplotlib.pyplot as pp
import matplotlib.cm as cm
import scanpy as sc

# Import data
data = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)

#Filter useless data
sc.pp.filter_genes(data, min_cells=1)
filtered_data = np.array(data._X.todense())

#Create Centroid Array
clusters_amount = 5
centroids_numbers = np.random.randint(2700, size=clusters_amount)
centroids_array = []
i = 0
while i < clusters_amount:
    centroids_array.append(centroids_array, filtered_data[centroids_numbers[i]], axis=1)
