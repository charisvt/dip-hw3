import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from functions import image_to_graph, n_cuts, calculate_n_cut_value

# Load the .mat file
data = loadmat('dip_hw_3.mat')
images = {'d2a': data['d2a'], 'd2b': data['d2b']}

# Normalize images to [0, 1] if they are not already
for name, img in images.items():
    if img.dtype == np.uint8:
        images[name] = img.astype(float) / 255.0

# Process each image
for name, img in images.items():
    print(f"Processing image: {name} for a single 2-way n-cut")
    
    # 1. Convert image to graph
    print("Converting image to graph...")
    affinity_mat = image_to_graph(img)
    
    # 2. Perform a single 2-way cut (k=2)
    print("Performing a 2-way n-cut...")
    labels = n_cuts(affinity_mat, k=2)
    
    # 3. Calculate the Ncut value for this partition
    n_cut_val = calculate_n_cut_value(affinity_mat, labels)
    print(f"Ncut value for this partition: {n_cut_val:.4f}")
    
    # 4. Visualize the result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Single 2-Way N-Cut for {name}')
    
    # Original image
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Segmented image
    segmented_image = labels.reshape(img.shape[0], img.shape[1])
    ax2.imshow(segmented_image, cmap='viridis')
    ax2.set_title(f'Segmented (Ncut = {n_cut_val:.4f})')
    ax2.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("Demo 3b complete.")
