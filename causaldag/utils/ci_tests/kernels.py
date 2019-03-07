import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def rbf_kernel(mat, precision):
    return np.exp(-precision/2 * euclidean_distances(mat, squared=True))


def delta_kernel(vec):
    if vec.ndim == 1:
        vec = vec.reshape(len(vec), 1)
    return (vec == vec.T).astype(int)


def center(m):
    n = m.shape[0]
    c = m - m.sum(axis=0)/n - m.sum(axis=1)[:, np.newaxis]/n + m.sum()/n**2  # centralized kernel matrix
    return c


if __name__ == '__main__':
    m = np.random.rand(3, 3)
    center(m)
    n = m.shape[0]