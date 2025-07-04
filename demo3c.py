import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from functions import image_to_graph, n_cuts_recursive

# --- Final Demonstration of Recursive N-Cuts ---

# Load the .mat file
data = loadmat('dip_hw_3.mat')
images = {'d2a': data['d2a'], 'd2b': data['d2b']}

# Define optimal thresholds found during debugging for each image
# T1 is the minimum size for a partition to be split
T1 = 5
# T2 is the Ncut value threshold, specific to each image's complexity
thresholds = {
    'd2a': 1.0,
    'd2b': 1.4
}

# Normalize images to [0, 1] if they are not already
for name, img in images.items():
    if img.dtype == np.uint8:
        images[name] = img.astype(float) / 255.0

# Process each image with its optimal thresholds
for name, img in images.items():
    T2 = thresholds[name]
    print(f"--- Processing image: {name} ---")
    print(f"Using thresholds: T1={T1}, T2={T2}")

    # 1. Convert image to graph
    print("Converting image to graph...")
    affinity_mat = image_to_graph(img)

    # 2. Perform recursive n-cuts
    print("Performing recursive n-cuts...")
    labels = n_cuts_recursive(affinity_mat, T1=T1, T2=T2)
    
    num_clusters = len(np.unique(labels))
    print(f"Found {num_clusters} clusters for image {name}.")

    # 3. Visualize the result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Recursive N-Cuts for {name} (Found {num_clusters} clusters)')

    # Original image
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Segmented image
    segmented_image = labels.reshape(img.shape[0], img.shape[1])
    ax2.imshow(segmented_image, cmap='viridis')
    ax2.set_title('Segmented Result')
    ax2.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\nDemo 3c complete.")
