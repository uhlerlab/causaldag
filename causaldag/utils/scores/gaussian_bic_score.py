import numpy as np
import numba

@numba.jit
def numba_inv(A):
    return np.linalg.inv(A)


def local_gaussian_bic_score(node, parents, suffstat, lambda_=None):
    n = suffstat["n"]
    lambda_ = lambda_ if lambda_ is not None else .5 * np.log2(n)
    C, i, p = suffstat['C'], node, list(parents)

    var = C[i, i] if not p else C[i, i] - C[i, p] @ numba_inv(C[np.ix_(p, p)]) @ C[p, i]
    log_prob = -.5*n*(1 + np.log2(var/n))
    penalty_term = lambda_*(1 + len(parents))

    return log_prob - penalty_term




