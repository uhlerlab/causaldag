from typing import Any, Union, List
from pygam import GAM
import numpy as np
from . import kernels


def hsic_test_vector(x: np.ndarray, y: np.ndarray, sig: float=1, ncol=100, alpha=0.05):
    raise NotImplementedError
    n = x.shape[0]
    H = np.eye(n) - np.ones([n, n])/n
    kernel_precision = 1/(sig**2)

    # === COMPUTE CENTRALIZED KERNEL MATRICES
    kx = kernels.rbf_kernel(x, kernel_precision)
    ky = kernels.rbf_kernel(y, kernel_precision)
    kx = H @ kx @ H
    ky = H @ ky @ H

    # === COMPUTE STATISTIC
    statistic = 1/n**2 * np.sum(kx * ky.T)

    critval = None
    p_value = None
    return dict(statistic=statistic, critval=critval, p_value=p_value, reject=statistic > critval)


def hsic_test(
        suffstat: Any,
        i: int,
        j: int,
        cond_set: Union[List[int], int]=None,
):
    raise NotImplementedError
    if cond_set is not None:
        g = GAM()
        residuals_i = g.deviance_residuals(suffstat[:, cond_set], suffstat[:, i])
        residuals_j = g.deviance_residuals(suffstat[:, cond_set], suffstat[:, j])
        return hsic_test_vector(residuals_i, residuals_j)


