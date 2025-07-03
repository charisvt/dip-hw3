from scipy.io import loadmat
from functions import spectral_clustering
import numpy as np

# Load the .mat file
data = loadmat('dip_hw_3.mat')

# Get the affinity matrix
affinity_matrix = data["d1a"]

# Define the values of k for the experiments
k_values = [2, 3, 4]

print("Running spectral clustering for d1a affinity matrix...")

# Run spectral clustering for each value of k
for k in k_values:
    print(f"--- k = {k} ---")
    labels = spectral_clustering(affinity_matrix, k)
    print("Cluster labels:")
    print(labels)
    print("-" * (13 + len(str(k))))


