import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import euclidean_distances
import itertools as itr
from scipy.stats import gamma


def rbf_kernel(mat, precision):
    return np.exp(-precision/2 * euclidean_distances(mat, squared=True))


def kci(
        Y: np.array,
        E: np.array,
        X: np.array,
        width:float=0.,
        alpha:float=0.05,
        unbiased:bool=False,
        gamma_approx:bool=False,
        n_draws:int=500,
        lam:float=1e-3,
        thresh:float=1e-5,
        num_eig:int=0,
):
    """

    :param Y: (n*d) matrix
    :param E: (n*d) matrix
    :param X: (n*d) matrix
    :param width: Kernel width. If 0, chosen automatically.
    :param alpha: Significance level
    :param unbiased: Whether bias correction should be applied.
    :param gamma_approx: If True, approximate the null distribution by a Gamma distribution. Otherwise, use a Monte
        Carlo approximation.
    :param n_draws: Number of draws in Monte Carlo approach if gamma_approx=False
    :param lam: Regularization parameter for matrix inversions
    :param thresh: Lower threshold for eigenvalues
    :return:
    """
    # ASSIGN VARIABLES USED THROUGHOUT METHOD
    n, d = X.shape
    Y = scale(Y)
    X = scale(X)
    E = scale(E)
    if width == 0:
        if n <= 200:
            width = .8
        elif n < 1200:
            width = .5
        else:
            width = .3
    if num_eig == 0:
        num_eig = n
    kernel_precision = 1/(width**2 * d)

    # === CREATE KERNEL MATRICES ===
    H = np.eye(n) - np.ones([n, n])/n

    kyx = rbf_kernel(np.concatenate((Y, X/2), axis=1), kernel_precision)
    kyx = H @ kyx @ H

    ke = rbf_kernel(E, kernel_precision)
    ke = H @ ke @ H

    kx = rbf_kernel(X, kernel_precision)
    kx = H @ kx @ H

    rx = np.eye(n) - kx @ np.linalg.inv(kx + lam * np.eye(n))
    kyx = rx @ kyx @ rx.T  # Equation (11)
    kex = rx @ ke @ rx.T  # Equation (12)

    statistic = np.sum(kyx * kex.T)
    dfE = np.sum(np.diag(np.eye(n) - rx))

    # === CALCULATE EIGENVALUES AND EIGENVECTORS ===
    eigvals_kyx, eigvecs_kyx, _ = np.linalg.svd((kyx + kyx.T)/2)
    eigvals_kyx = eigvals_kyx[:num_eig]
    eigvecs_kyx = eigvecs_kyx[:, :num_eig]
    eigvals_kex, eigvecs_kex, _ = np.linalg.svd((kex + kex.T) / 2)
    eigvals_kex = eigvals_kex[:num_eig]
    eigvecs_kex = eigvecs_kex[:, :num_eig]

    # === THRESHOLD EIGENVALUES AND EIGENVECTORS ===
    ixs_yx = eigvals_kyx > np.max(eigvals_kyx)*thresh
    eigvals_kyx = eigvals_kyx[ixs_yx]
    eigvecs_kyx = eigvecs_kyx[:, :ixs_yx]
    ixs_ex = eigvals_kex > np.max(eigvals_kex) * thresh
    eigvals_kex = eigvals_kex[ixs_ex]
    eigvecs_kex = eigvecs_kex[:, :ixs_ex]

    # === CALCULATE PRODUCT OF EIGENVECTORS WITH SQUARE ROOT OF EIGENVALUES
    eigprod_kyx = eigvecs_kyx * np.sqrt(eigvals_kyx)[None, :]  # TODO: CHECK
    eigprod_kex = eigvecs_kex * np.sqrt(eigvals_kex)[None, :]  # TODO: CHECK

    # === CALCULATE W ===
    d_yx = eigprod_kyx.shape[1]
    d_ex = eigprod_kex.shape[1]

    w = np.zeros(d_yx*d_ex, n)
    for i, j in itr.product(range(d_yx), range(d_ex)):
        w[(i-1)*d_ex+j] = eigprod_kyx[:, i] * eigprod_kex[:, j]  # TODO: CHECK
    ww = w @ w.T if d_yx*d_ex < n else w.T @ w

    if not gamma_approx:
        # TODO: CHECK
        pass
    else:
        mean_approx = np.sum(np.diag(ww))
        var_approx = 2*np.sum(np.diag(ww**2))
        k_approx = mean_approx**2/var_approx
        prec_approx = var_approx/mean_approx

        pval = gamma.cdf(statistic, k_approx, scale=prec_approx)

    return statistic, pval






