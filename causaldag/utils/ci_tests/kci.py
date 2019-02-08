import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import euclidean_distances
import itertools as itr
from scipy.stats import gamma
from typing import Dict, Union, List, Optional
from causaldag.utils.ci_tests import kernels
import pygam
from causaldag.utils.ci_tests._utils import residuals, combined_mat


def ki_test_vector(
        Y: np.ndarray,
        X: np.ndarray,
        width_x: float=0.,
        width_y: float=0.,
        alpha: float=0.05,
        gamma_approx: bool=True,
        n_draws: int=500,
        lam: float=1e-3,
        thresh: float=1e-5,
        num_eig: int=0,
        catgorical_x: bool=False
):
    """
    Test the null hypothesis that Y and X are independent

    :param Y: (n*_) matrix
    :param X: (n*_) matrix
    :param width_x: Kernel width. If 0, chosen automatically.
    :param width_y: Kernel width. If 0, chosen automatically.
    :param alpha: Significance level
    :param gamma_approx: If True, approximate the null distribution by a Gamma distribution. Otherwise, use a Monte
        Carlo approximation.
    :param n_draws:
    :param lam:
    :param thresh:
    :param num_eig:
    :return:
    """
    # === ASSIGN VARIABLES USED THROUGHOUT METHOD ===
    if X.ndim == 1:
        X = X.reshape((len(X), 1))
    if Y.ndim == 1:
        Y = Y.reshape((len(Y), 1))
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("Y should have the same number of samples as X")

    # === CREATE KERNEL MATRICES ===
    if catgorical_x:
        kx = kernels.delta_kernel(X)
    else:
        if width_x == 0:
            width_x = np.median(euclidean_distances(X))
        X = scale(X)
        kernel_precision_x = 1 / (width_x ** 2)  # TODO: CHECK
        kx = kernels.rbf_kernel(X, kernel_precision_x)
    if width_y == 0:
        width_y = np.median(euclidean_distances(Y))
    Y = scale(Y)
    kernel_precision_y = 1/(width_y ** 2)  # TODO: CHECK

    H = np.eye(n) - np.ones([n, n])/n
    kx = H @ kx @ H
    ky = kernels.rbf_kernel(Y, kernel_precision_y)
    ky = H @ ky @ H

    # === COMPUTE STATISTIC ====
    statistic = np.sum(kx * ky.T)/n  # same as trace of product

    # === COMPUTE NULL DISTRIBUTION ====
    if not gamma_approx:
        raise NotImplementedError
    else:
        mean_approx = 1/n**2 * np.trace(kx) * np.trace(ky)
        var_approx = 2/n**4 * np.sum(kx * kx) * np.sum(ky * ky)
        # k is shape, theta is scale
        k_approx = mean_approx**2/var_approx
        prec_approx = var_approx/mean_approx

        critval = gamma.ppf(1-alpha, k_approx, scale=prec_approx)
        pval = 1 - gamma.cdf(statistic, k_approx, scale=prec_approx)

    return dict(statistic=statistic, critval=critval, p_value=pval, reject=statistic>critval)


def kci_test_vector(
        Y: np.array,
        E: np.array,
        X: np.array,
        width: float=0.,
        alpha: float=0.05,
        unbiased: bool=False,
        gamma_approx: bool=True,
        n_draws: int=500,
        lam: float=1e-3,
        thresh: float=1e-5,
        num_eig: int=0,
        catgorical_e: bool=False
) -> Dict:
    """
    Test the null hypothesis that Y and E are independent given X.

    :param Y: (n*_) matrix
    :param E: (n*_) matrix
    :param X: (n*d) matrix
    :param width: Kernel width. If 0, chosen automatically.
    :param alpha: Significance level
    :param unbiased: Whether bias correction should be applied.
    :param gamma_approx: If True, approximate the null distribution by a Gamma distribution. Otherwise, use a Monte
        Carlo approximation.
    :param n_draws: Number of draws in Monte Carlo approach if gamma_approx=False
    :param lam: Regularization parameter for matrix inversions
    :param thresh: Lower threshold for eigenvalues
    :return: (statistic, critval, pval). The p-value for the null hypothesis that Y and E are independent given X.
    """
    # ASSIGN VARIABLES USED THROUGHOUT METHOD
    if X.ndim == 1:
        X = X.reshape((len(X), 1))
    if Y.ndim == 1:
        Y = Y.reshape((len(Y), 1))
    if E.ndim == 1:
        E = E.reshape((len(E), 1))
    n, d = X.shape
    if Y.shape[0] != n:
        raise ValueError("Y should have the same number of samples as X")
    if E.shape[0] != n:
        raise ValueError("E should have the same number of samples as X and Y")

    Y = scale(Y)
    X = scale(X)
    if width == 0:
        if n <= 200:
            width = 0.8
        elif n < 1200:
            width = 0.5
        else:
            width = 0.3
    if num_eig == 0:
        num_eig = n
    kernel_precision = 1/(width**2 * d)

    if catgorical_e:
        ke = kernels.delta_kernel(E)
    else:
        E = scale(E)
        ke = kernels.rbf_kernel(E, kernel_precision)

    # === CREATE KERNEL MATRICES ===
    H = np.eye(n) - np.ones([n, n])/n

    kyx = kernels.rbf_kernel(np.concatenate((Y, X/2), axis=1), kernel_precision)
    kyx = H @ kyx @ H  # Centralize Kyx

    ke = H @ ke @ H  # Centralize Ke

    kx = kernels.rbf_kernel(X, kernel_precision)
    kx = H @ kx @ H  # Centralize Kx

    rx = np.eye(n) - kx @ np.linalg.inv(kx + lam * np.eye(n))
    kyx = rx @ kyx @ rx.T  # Equation (11)
    kex = rx @ ke @ rx.T  # Equation (12)

    statistic = np.sum(kyx * kex.T)
    dfE = np.sum(np.diag(np.eye(n) - rx))

    # === CALCULATE EIGENVALUES AND EIGENVECTORS ===
    eigvecs_kyx, eigvals_kyx, _ = np.linalg.svd((kyx + kyx.T)/2)
    eigvals_kyx = eigvals_kyx[:num_eig]
    eigvecs_kyx = eigvecs_kyx[:, :num_eig]
    eigvecs_kex, eigvals_kex, _ = np.linalg.svd((kex + kex.T) / 2)
    eigvals_kex = eigvals_kex[:num_eig]
    eigvecs_kex = eigvecs_kex[:, :num_eig]

    # === THRESHOLD EIGENVALUES AND EIGENVECTORS ===
    ixs_yx = eigvals_kyx > np.max(eigvals_kyx)*thresh
    eigvals_kyx = eigvals_kyx[ixs_yx]
    eigvecs_kyx = eigvecs_kyx[:, ixs_yx]
    ixs_ex = eigvals_kex > np.max(eigvals_kex) * thresh
    eigvals_kex = eigvals_kex[ixs_ex]
    eigvecs_kex = eigvecs_kex[:, ixs_ex]

    # === CALCULATE PRODUCT OF EIGENVECTORS WITH SQUARE ROOT OF EIGENVALUES
    eigprod_kyx = eigvecs_kyx * np.sqrt(eigvals_kyx)[None, :]  # TODO: CHECK
    eigprod_kex = eigvecs_kex * np.sqrt(eigvals_kex)[None, :]  # TODO: CHECK

    # === CALCULATE W ===
    d_yx = eigprod_kyx.shape[1]
    d_ex = eigprod_kex.shape[1]

    w = np.zeros([d_yx*d_ex, n])
    for i, j in itr.product(range(d_yx), range(d_ex)):
        w[(i-1)*d_ex+j] = eigprod_kyx[:, i] * eigprod_kex[:, j]  # TODO: CHECK
    ww = w @ w.T if d_yx*d_ex < n else w.T @ w

    if not gamma_approx:
        # TODO
        raise NotImplementedError
    else:
        mean_approx = np.sum(np.diag(ww))
        var_approx = 2*np.sum(np.diag(ww**2))
        k_approx = mean_approx**2/var_approx
        prec_approx = var_approx/mean_approx

        critval = gamma.ppf(1-alpha, k_approx, scale=prec_approx)
        pval = 1 - gamma.cdf(statistic, k_approx, scale=prec_approx)

    return dict(statistic=statistic, critval=critval, p_value=pval, reject=statistic>critval)


def kci_test(
        suffstat: np.array,
        i,
        j,
        cond_set: Union[List[int], int]=None,
        width: float=0.,
        alpha: float=0.05,
        unbiased: bool=False,
        gamma_approx: bool=True,
        regress: bool=True,
        n_draws: int=500,
        lam: float=1e-3,
        thresh: float=1e-5,
        num_eig: int=0,
        categorical_e: bool=False
) -> Dict:
    if isinstance(cond_set, int):
        cond_set = [cond_set]
    if cond_set is None or len(cond_set) == 0:
        return ki_test_vector(
            suffstat[:, i],
            suffstat[:, j],
            width_x=width,
            width_y=width,
            alpha=alpha,
            gamma_approx=gamma_approx,
            n_draws=n_draws,
            lam=lam,
            thresh=thresh,
            num_eig=num_eig,
            catgorical_x=categorical_e
        )
    else:
        if regress:
            residuals_i, residuals_j = residuals(suffstat, i, j, cond_set)
            return ki_test_vector(
                residuals_i,
                residuals_j,
                width_x=width,
                width_y=width,
                alpha=alpha,
                gamma_approx=gamma_approx,
                n_draws=n_draws,
                lam=lam,
                thresh=thresh,
                num_eig=num_eig,
                catgorical_x=categorical_e
            )
        else:
            return kci_test_vector(
                suffstat[:, i],
                suffstat[:, j],
                suffstat[:, cond_set],
                width=width,
                alpha=alpha,
                unbiased=unbiased,
                gamma_approx=gamma_approx,
                n_draws=n_draws,
                lam=lam,
                thresh=thresh,
                num_eig=num_eig,
                catgorical_e=categorical_e
            )


def kci_invariance_test(
        samples1: np.ndarray,
        samples2: np.ndarray,
        i: int,
        cond_set: Optional[Union[List[int], int]]=None,
        width: float=0,
        alpha: float=0.05,
        unbiased: bool=False,
        regress: bool=True,
        gamma_approx: bool=True,
        n_draws: int=500,
        lam: float=1e-3,
        thresh: float=1e-5,
        num_eig: int=0,
):
    if isinstance(cond_set, int):
        cond_set = [cond_set]
    if cond_set is None:
        cond_set = []

    mat = combined_mat(samples1, samples2, i, cond_set)
    return kci_test(
        mat, 0, 1, list(range(2, 2+len(cond_set))),
        width=width,
        alpha=alpha,
        unbiased=unbiased,
        gamma_approx=gamma_approx,
        regress=regress,
        n_draws=n_draws,
        lam=lam,
        thresh=thresh,
        num_eig=num_eig,
    )
    # i_values = np.concatenate((samples1[:, i], samples2[:, i]))
    # labels = np.concatenate((np.zeros(samples1.shape[0]), np.ones(samples2.shape[0])))
    # if cond_set is None or len(cond_set) == 0:
    #     return ki_test_vector(
    #         i_values,
    #         labels,
    #         width_x=width,
    #         width_y=width,
    #         alpha=alpha,
    #         gamma_approx=gamma_approx,
    #         n_draws=n_draws,
    #         lam=lam,
    #         thresh=thresh,
    #         num_eig=num_eig,
    #         catgorical_x=True
    #     )
    # else:
    #     cond_set_values = np.concatenate((samples1[:, cond_set], samples2[:, cond_set]))
    #     return kci_test_vector(
    #         i_values,
    #         labels,
    #         cond_set_values,
    #         width=width,
    #         alpha=alpha,
    #         unbiased=unbiased,
    #         gamma_approx=gamma_approx,
    #         n_draws=n_draws,
    #         lam=lam,
    #         thresh=thresh,
    #         num_eig=num_eig,
    #         catgorical_e=True
    #     )





