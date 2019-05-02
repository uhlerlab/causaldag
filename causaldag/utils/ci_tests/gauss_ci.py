from typing import Dict
from math import erf
import numba
from numpy import sqrt, log1p, abs, ix_, diag, corrcoef
from numpy.linalg import inv
from . import MemoizedCI_Tester


@numba.jit
def numba_inv(A):
    return inv(A)


def gauss_ci_suffstat(samples, invert=True):
    """
    Helper function to compute the sufficient statistics for the gauss_ci_test from data.

    Parameters
    ----------
    samples:
        (n x p) matrix, where n is the number of samples and p is the number of variables.
    invert:
        if True, compute the inverse correlation matrix, and normalize it into the partial correlation matrix. This
        will generally speed up the gauss_ci_test if large conditioning sets are used.

    Return
    ------
    dictionary of sufficient statistics
    """
    n = samples.shape[0]
    C = corrcoef(samples, rowvar=False)
    if invert:
        K = numba_inv(C)
        rho = K/sqrt(diag(K))/sqrt(diag(K))[:, None]
        return dict(C=C, n=n, K=K, rho=rho)
    return dict(C=C, n=n)


def gauss_ci_test(suffstat: Dict, i, j, cond_set=None, alpha=0.01):
    """
    Test the null hypothesis that i and j are conditionally independent given cond_set via Fisher's z-transform.

    Parameters
    ----------
    suffstat:
        dictionary containing:
        'n' -- number of samples
        'C' -- correlation matrix
        'K' (optional) -- inverse correlation matrix
        'rho' (optional) -- partial correlation matrix (K, normalized so diagonals are 1).
    i:
        position of first variable in correlation matrix.
    j:
        position of second variable in correlation matrix.
    cond_set:
        positions of conditioning set in correlation matrix.
    alpha:
        Significance level.

    Return
    ------
    dictionary containing statistic, p_value, and reject.
    """
    n = suffstat['n']
    C = suffstat.get('C')
    p = C.shape[0]
    rho = suffstat.get('rho')
    K = suffstat.get('K')
    n_cond = 0 if cond_set is None else len(cond_set)

    # === COMPUTE PARTIAL CORRELATION
    # partial correlation is correlation if there is no conditioning
    if cond_set is None or len(cond_set) == 0:
        r = C[i, j]
    # used closed-form
    elif len(cond_set) == 1:
        k = list(cond_set)[0]
        r = (C[i, j] - C[i, k]*C[j, k]) / sqrt((1 - C[j, k]**2) * (1 - C[i, k]**2))
    # when conditioning on everything, partial correlation comes from normalized precision matrix
    elif len(cond_set) == p - 2 and rho is not None:
        r = -rho[i, j]
    # faster to use Schur complement if conditioning set is large and precision matrix is pre-computed
    elif len(cond_set) >= p/2 and K is not None:
        rest = list(set(range(C.shape[0])) - {i, j, *cond_set})

        if len(rest) == 1:
            theta_ij = K[ix_([i, j], [i, j])] - K[ix_([i, j], rest)] @ K[ix_(rest, [i, j])] / K[rest[0], rest[0]]
        else:
            theta_ij = K[ix_([i, j], [i, j])] - K[ix_([i, j], rest)] @ numba_inv(K[ix_(rest, rest)]) @ K[ix_(rest, [i, j])]
        r = -theta_ij[0, 1] / sqrt(theta_ij[0, 0] * theta_ij[1, 1])
    else:
        theta = numba_inv(C[ix_([i, j, *cond_set], [i, j, *cond_set])])
        r = -theta[0, 1]/sqrt(theta[0, 0] * theta[1, 1])

    # === COMPUTE STATISTIC AND P-VALUE
    # note: log1p(2r/(1-r)) = log((1+r)/(1-r)) but is more numerically stable for r near 0
    statistic = sqrt(n - n_cond - 3) * abs(.5 * log1p(2*r/(1 - r)))
    # note: erf is much faster than norm.cdf
    p_value = 1 - .5*(1 + erf(statistic/sqrt(2)))

    return dict(statistic=statistic, p_value=p_value, reject=p_value < alpha)


class MemoizedGaussCI_Tester(MemoizedCI_Tester):
    def __init__(self, suffstat: Dict, track_times=False, detailed=False, **kwargs):
        MemoizedCI_Tester.__init__(self, gauss_ci_test, suffstat, track_times=track_times, detailed=detailed)


