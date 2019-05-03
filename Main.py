# Import libraries
import numpy
import pandas
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
import scanpy as sc

# Import data
data = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)
# raw_data = data._X.todense() # returns matrix

#Filter useless data
sc.pp.filter_genes(data, min_cells=1)
filtered_data = np.array(data._X.todense())

#Create Centroid Array
clusters_amount = 5
centroids_numbers = np.random.randint(2700, size=clusters_amount)
centroids_array = np.empty([0, 16634])
i = 0
while i < clusters_amount:
    randompatient = centroids_numbers[i]
    centroids_array = np.append(centroids_array, [filtered_data[randompatient, :]], axis = 0)
    i += 1

print(centroids_array)
