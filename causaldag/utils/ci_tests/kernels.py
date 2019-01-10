import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def rbf_kernel(mat, precision):
    return np.exp(-precision/2 * euclidean_distances(mat, squared=True))


def delta_kernel(vec):
    if vec.ndim == 1:
        vec = vec.reshape(len(vec), 1)
    return (vec == vec.T).astype(int)
