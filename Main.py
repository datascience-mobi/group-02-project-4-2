# Import libraries
import numpy as np
import pandas
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
import scanpy as sc
from datetime import datetime

# Global Variables
t1 = 0
patients = 0
genes = 0
k = 0

# Functions

# Define distance function which takes integer inputs which identify patient and centroid
def dist(patient_point, cluster_number):
    a = filtered_data[patient_point, :]
    b = centroids_array[cluster_number, :]
    dist = np.linalg.norm(a-b)
    return dist

def runtime_start():
    global t1 
    t1 = datetime.now().time()

def runtime_end():
    t2 = datetime.now().time()
    FMT = '%H:%M:%S.%f'
    elapsed = str(datetime.strptime(str(t2), FMT) - datetime.strptime(str(t1), FMT))
    return elapsed

# Import data
runtime_start()
data = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)

# Filter useless data
sc.pp.filter_genes(data, min_cells=1)
filtered_data = np.array(data._X.todense())

# Create Centroid Array by randomly picking 5 patients from data  
k = 5
patients = filtered_data.shape[0]
genes = filtered_data.shape[1]
centroids_numbers = np.random.randint(patients, size=k)
centroids_array = np.empty([0, genes])
i = 0


while i < k:
    randompatient = centroids_numbers[i]
    centroids_array = np.append(centroids_array, [filtered_data[randompatient, :]], axis = 0)
    i += 1

# Loop über alle Patienten
i = 0
nearest_centroid = np.zeros([patients, 1])    
while (i < patients):
    sml_distance = 0

    # While loop selecting every centroid
    j = 0
    while (j < k):

        if sml_distance == 0 or dist(i,j) < sml_distance:
            sml_distance = dist(i,j)
            nearest_centroid[i, 0] = j
        j += 1
    i += 1

runtime_end()