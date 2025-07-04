import numpy as np

def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    """
    Converts an image to a fully connected graph represented by an affinity matrix.

    Each pixel in the image is a node in the graph. The weight of the edge
    between two pixels is calculated based on the Euclidean distance of their
    channel values (luminosity).

    Args:
        img_array (np.ndarray): A NumPy array of shape (M, N, C), where M and N
            are the image dimensions and C is the number of channels.
            The values should be floats in the range [0, 1].

    Returns:
        np.ndarray: An affinity matrix of shape (M*N, M*N) representing the
            fully connected graph. The affinity is calculated as 1/exp(d(i,j) == exp(-d(i,j)),
            where d(i,j) is the Euclidean distance between the channel vectors
            of pixel i and pixel j.
    """
    # Get image dimensions
    m, n, c = img_array.shape

    # Reshape the image array to a list of pixels
    # Each row is a pixel, and columns are the channel values
    pixel_list = img_array.reshape(-1, c)

    # Calculate the pairwise Euclidean distances between all pixels
    p1 = pixel_list[:, np.newaxis, :]
    p2 = pixel_list[np.newaxis, :, :]
    diff = p1 - p2
    dist_sq = np.sum(diff**2, axis=-1)
    distances = np.sqrt(dist_sq)

    # Calculate the affinity matrix using the formula A[i,j] = exp(-d(i,j))
    affinity_map = np.exp(-distances)

    return affinity_map


from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    """
    Performs spectral clustering on a given affinity matrix.

    Args:
        affinity_mat : The affinity matrix (W) of the graph.
        k : The number of clusters to form.

    Returns:
        np.ndarray: A 1-D array of length M*N with the cluster labels for each node.
    """
    # Calculate the Laplacian matrix L = D - W
    D = np.diag(np.sum(affinity_mat, axis=1))
    L = D - affinity_mat

    # Solve the eigenvalue problem for the k smallest eigenvalues
    # We ask for k+1 eigenvectors and discard the first one (trivial eigenvector)
    # 'SM' for smallest magnitude eigenvalues
    eigenvalues, eigenvectors = eigs(L, k=k, which='SM')

    # The eigenvectors are in the columns of the returned array
    # We need to take the real part as they can be complex
    U = np.real(eigenvectors)

    # Apply K-Means clustering
    # Each row of U is a data point to be clustered
    kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
    cluster_idx = kmeans.fit_predict(U)

    return cluster_idx


def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    """
    Performs image segmentation using the non-recursive Normalized Cuts algorithm.

    This method solves the generalized eigenvalue problem (D-W)x = lambda*Dx
    to find the k smallest eigenvectors, which are then used for clustering.

    Args:
        affinity_mat (np.ndarray): The affinity matrix (W) of the graph.
        k (int): The number of clusters to form.

    Returns:
        np.ndarray: A 1-D array with the cluster labels for each node.
    """
    # Calculate the diagonal matrix D
    D = np.diag(np.sum(affinity_mat, axis=1))

    # Calculate the Laplacian matrix L
    L = D - affinity_mat

    # Solve the generalized eigenvalue problem L*x = l*D*x for the k smallest eigenvalues.
    # 'SM' - Smallest Magnitude.
    try:
        eigenvalues, eigenvectors = eigs(L, k=k, M=D, which='SM')
    except Exception as e:
        # Add a small value to the diagonal of D to avoid singularity issues
        D_reg = D + np.eye(D.shape[0]) * 1e-9
        eigenvalues, eigenvectors = eigs(L, k=k, M=D_reg, which='SM')

    # The eigenvectors are in the columns, take the real part
    U = np.real(eigenvectors)

    # Cluster the rows of U using K-Means
    kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
    cluster_idx = kmeans.fit_predict(U)

    return cluster_idx

def calculate_n_cut_value(full_W: np.ndarray, partition_nodes_indices: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the Ncut value for a bipartition of a graph.

    Args:
        full_W (np.ndarray): The affinity matrix of the original, full graph.
        partition_nodes_indices (np.ndarray): The indices of the nodes in the current partition.
        labels (np.ndarray): A 1-D array of cluster labels (0 or 1) for the partition.

    Returns:
        float: The calculated Ncut value.
    """
    cluster_A_local_indices = np.where(labels == 0)[0]
    cluster_B_local_indices = np.where(labels == 1)[0]

    if len(cluster_A_local_indices) == 0 or len(cluster_B_local_indices) == 0:
        return float('inf')  # Invalid cut

    # Map local indices to global indices
    cluster_A_global_indices = partition_nodes_indices[cluster_A_local_indices]
    cluster_B_global_indices = partition_nodes_indices[cluster_B_local_indices]

    # Calculate associations using the full affinity matrix
    assoc_A_A = full_W[np.ix_(cluster_A_global_indices, cluster_A_global_indices)].sum()
    assoc_B_B = full_W[np.ix_(cluster_B_global_indices, cluster_B_global_indices)].sum()
    assoc_A_V = full_W[cluster_A_global_indices, :].sum()
    assoc_B_V = full_W[cluster_B_global_indices, :].sum()

    term1 = assoc_A_A / assoc_A_V if assoc_A_V != 0 else 0
    term2 = assoc_B_B / assoc_B_V if assoc_B_V != 0 else 0
    n_assoc = term1 + term2

    return 2 - n_assoc

def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float) -> np.ndarray:
    """
    Performs recursive normalized cuts on an affinity matrix using an iterative approach.

    Args:
        affinity_mat (np.ndarray): The affinity matrix of the graph.
        T1 (int): The minimum size of a cluster. If a cluster is smaller than T1,
                  it will not be partitioned further.
        T2 (float): The Ncut value threshold. If the Ncut value of a partition
                    is greater than T2, the partition will not be split.

    Returns:
        np.ndarray: A 1-D array of cluster labels.
    """
    num_nodes = affinity_mat.shape[0]
    final_labels = np.zeros(num_nodes, dtype=int)
    
    # Use a queue for iterative processing of partitions
    partitions_to_process = [np.arange(num_nodes)]
    current_label = 0

    while partitions_to_process:
        nodes_indices = partitions_to_process.pop(0)

        # Base Case 1: Partition is too small
        if len(nodes_indices) < T1:
            final_labels[nodes_indices] = current_label
            current_label += 1
            continue

        sub_W = affinity_mat[np.ix_(nodes_indices, nodes_indices)]

        # Perform a 2-way cut
        try:
            partition_labels = n_cuts(sub_W, k=2)
            # If n_cuts returns a single cluster, stop splitting this partition
            if len(np.unique(partition_labels)) < 2:
                final_labels[nodes_indices] = current_label
                current_label += 1
                continue
        except Exception:
            # If cutting fails, treat it as a single segment
            final_labels[nodes_indices] = current_label
            current_label += 1
            continue

        # Base Case 2: Ncut value is too high
        n_cut_val = calculate_n_cut_value(affinity_mat, nodes_indices, partition_labels)
        if n_cut_val > T2:
            final_labels[nodes_indices] = current_label
            current_label += 1
            continue

        # If checks pass, add the two new sub-partitions to the queue for further processing
        nodes_A = nodes_indices[partition_labels == 0]
        nodes_B = nodes_indices[partition_labels == 1]
        partitions_to_process.append(nodes_A)
        partitions_to_process.append(nodes_B)

    # Post-process labels to be contiguous from 0
    unique_labels = np.unique(final_labels)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    final_labels = np.array([label_map[l] for l in final_labels])
    
    return final_labels


