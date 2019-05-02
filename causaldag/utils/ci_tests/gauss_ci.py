import numpy as np
from typing import Dict
# from scipy.stats import norm
from math import erf
import numba
from numpy import sqrt, log1p, abs, ix_
from numpy.linalg import inv
from . import MemoizedCI_Tester


@numba.jit
def numba_inv(A):
    return inv(A)


def gauss_ci_test(suffstat: Dict, i, j, cond_set=None, alpha=0.01):
    """
    Test the null hypothesis that i and j are conditionally independent given cond_set via Fisher's z-transform.

    Parameters
    ----------
    suffstat:
        dictionary containing 'n': number of samples, 'C': correlation matrix, and 'K': partial correlation matrix.
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
    K = suffstat.get('K')
    n_cond = 0 if cond_set is None else len(cond_set)

    # === COMPUTE PARTIAL CORRELATION
    if cond_set is None or len(cond_set) == 0:
        r = C[i, j]
    elif len(cond_set) == 1:
        k = list(cond_set)[0]
        r = (C[i, j] - C[i, k]*C[j, k]) / sqrt((1 - C[j, k]**2) * (1 - C[i, k]**2))
    elif len(cond_set) == C.shape[0] - 2 and K is not None:
        r = -K[i, j]
    else:
        theta = numba_inv(C[ix_([i, j, *cond_set], [i, j, *cond_set])])
        r = -theta[0, 1]/sqrt(theta[0, 0] * theta[1, 1])
    # print(r)

    statistic = sqrt(n - n_cond - 3) * abs(.5 * log1p(2*r/(1 - r)))
    # NOTE: log1p(2r/(1-r)) = log((1+r)/(1-r)) but is more numerically stable for r near 0

    # crit_val = norm.ppf(1 - alpha/2)
    # p_value = 1 - norm.cdf(statistic)
    p_value = 1 - .5*(1 + erf(statistic/sqrt(2)))  # much faster than norm.cdf

    return dict(statistic=statistic, p_value=p_value, reject=p_value < alpha)


class MemoizedGaussCI_Tester(MemoizedCI_Tester):
    def __init__(self, suffstat: Dict, track_times=False, detailed=False, **kwargs):
        MemoizedCI_Tester.__init__(self, gauss_ci_test, suffstat, track_times=track_times, detailed=detailed)


