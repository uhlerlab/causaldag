from typing import Union, List, Dict
import numpy as np
from causaldag.utils.ci_tests import kernels
from scipy.stats import gamma
from causaldag.utils.ci_tests._utils import residuals
from causaldag.utils.core_utils import to_list
from scipy.special import gdtr
import ipdb


def hsic_test_vector(
        x: np.ndarray,
        y: np.ndarray,
        sig: float=1/np.sqrt(2),
        alpha=0.05
) -> Dict:
    """
    Test for independence of X and Y using the Hilbert-Schmidt Information Criterion.

    Parameters
    ----------
    x:
        vector of samples from X.
    y:
        vector of samples from Y.
    sig:
        width parameter.
    alpha:
        significance level.

    Returns
    -------

    """
    if x.ndim == 1:
        x = x.reshape((len(x), 1))
    if y.ndim == 1:
        y = y.reshape((len(y), 1))
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("Y should have the same number of samples as X")

    n = x.shape[0]
    kernel_precision = 1/(sig**2)

    # === COMPUTE CENTRALIZED KERNEL MATRICES
    kx = kernels.rbf_kernel_fast(x, kernel_precision)
    ky = kernels.rbf_kernel_fast(y, kernel_precision)
    kx_centered = kernels.center(kx)
    ky_centered = kernels.center(ky)
    # ipdb.set_trace()

    # === COMPUTE STATISTIC
    statistic = 1/n**2 * np.sum(kx_centered * ky_centered.T)  # SAME AS trace(kx_centered @ ky_centered)

    mu_x = 1/(n*(n-1)) * np.sum(kx - np.diag(np.diag(kx)))  # SUM OFF-DIAGONALS
    mu_y = 1/(n*(n-1)) * np.sum(ky - np.diag(np.diag(ky)))
    mean_approx = 1/n * (1 + mu_x*mu_y - mu_x - mu_y)
    var_approx = 2*(n-4)*(n-5)/(n*(n-1)*(n-2)*(n-3)) * np.sum(kx_centered * kx_centered.T) * np.sum(ky_centered * ky_centered.T) / n**4

    # NEW
    k_approx = mean_approx ** 2 / var_approx
    prec_approx = var_approx / mean_approx

    # critval = gamma.ppf(1-alpha, k_approx, scale=prec_approx)
    # p_value = 1 - gamma.cdf(statistic, k_approx, scale=prec_approx)
    p_value = 1 - gdtr(1/prec_approx, k_approx, statistic)
    # print(p_value, p_value_)

    return dict(
        statistic=statistic,
        # critval=critval,
        p_value=p_value,
        reject=p_value < alpha,
        mean_approx=mean_approx,
        var_approx=var_approx
    )


def hsic_test(
        suffstat: np.ndarray,
        i: int,
        j: int,
        cond_set: Union[List[int], int]=None,
        alpha: float=0.05
) -> Dict:
    """
    Test for (conditional) independence using the Hilbert-Schmidt Information Criterion. If a conditioning set is
    specified, first perform non-parametric regression, then test residuals.

    Parameters
    ----------
    suffstat:
        Matrix of samples.
    i:
        column position of first variable.
    j:
        column position of second variable.
    cond_set:
        column positions of conditioning set.
    alpha:
        Significance level of the test.

    Returns
    -------

    """
    cond_set = to_list(cond_set)
    if len(cond_set) == 0:
        return hsic_test_vector(suffstat[:, i], suffstat[:, j], alpha=alpha)
    else:
        residuals_i, residuals_j = residuals(suffstat, i, j, cond_set)
        return hsic_test_vector(residuals_i, residuals_j, alpha=alpha)


if __name__ == '__main__':
    import numpy as np
    from line_profiler import LineProfiler

    lp = LineProfiler()
    X1 = np.random.laplace(0, 1, size=1000)
    X2 = np.random.laplace(0, 1, size=1000)
    lp.add_function(hsic_test_vector)
    for _ in range(10):
        lp.runcall(hsic_test_vector, X1, X2)
    lp.print_stats()


