import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as rbf
from scipy.linalg.blas import sgemm
import numexpr as ne


def rbf_kernel_fast(X, precision):
    gamma = precision / 2
    X_norm = -gamma * np.einsum('ij,ij->i', X, X)
    return ne.evaluate('exp(A + B + C)', {
        'A': X_norm[:, None],
        'B': X_norm[None, :],
        'C': sgemm(alpha=2.0 * gamma, a=X, b=X, trans_b=True),
        'g': gamma,
    })


def rbf_kernel_basic(mat, precision):
    return rbf(mat, gamma=precision / 2)


def delta_kernel(vec):
    if vec.ndim == 1:
        vec = vec.reshape(len(vec), 1)
    return (vec == vec.T).astype(int)


def center(m):
    return m - m.mean(axis=0) - m.sum(axis=1)[:, None] + m.mean()


def center_fast_mutate(m):
    row_mean = m.mean(axis=0)[None, :]
    col_mean = m.mean(axis=1)[:, np.newaxis]
    s = m.mean()
    np.subtract(m, row_mean, m)
    np.subtract(m, col_mean, m)
    np.add(m, s, m)
    return m


if __name__ == '__main__':
    m = np.random.rand(3, 3)
    center(m)
    n = m.shape[0]
