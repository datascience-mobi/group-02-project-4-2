# Import libraries
import numpy
import pandas
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
import scanpy as sc

# Import data
data = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/', var_names='gene_symbols', cache=True)
# raw_data = data._X.todense() # returns matrix

