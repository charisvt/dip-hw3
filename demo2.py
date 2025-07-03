import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from functions import image_to_graph, spectral_clustering

# Load the .mat file
data = loadmat('dip_hw_3.mat')
images = {'d2a': data['d2a'], 'd2b': data['d2b']}
k_values = [2, 3, 4]

# Normalize images to [0, 1] if they are not already
for name, img in images.items():
    if img.dtype == np.uint8:
        images[name] = img.astype(float) / 255.0

# Process each image
for name, img in images.items():
    print(f"Processing image: {name}")
    
    # 1. Convert image to graph
    print("Converting image to graph...")
    affinity_mat = image_to_graph(img)
    
    # Create a figure to display results for the current image
    fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(15, 5))
    fig.suptitle(f'Spectral Clustering Results for {name}')
    
    # Display original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 2. Perform spectral clustering for each k
    for i, k in enumerate(k_values):
        print(f"Performing spectral clustering for k={k}...")
        labels = spectral_clustering(affinity_mat, k)
        
        # Reshape labels to image dimensions for visualization
        segmented_image = labels.reshape(img.shape[0], img.shape[1])
        
        # Display segmented image
        ax = axes[i + 1]
        im = ax.imshow(segmented_image, cmap='viridis')
        ax.set_title(f'k = {k}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("All experiments complete.")
