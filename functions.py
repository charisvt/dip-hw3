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
