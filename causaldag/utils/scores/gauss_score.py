import numpy as np
import numba

@numba.jit
def numba_inv(A):
    return np.linalg.inv(A)


def local_score(self, node, parents, suffstat, lambda_=None):
    lambda_ if lambda_ is not None else .5 * np.log2(self.n)
    C, i, p = suffstat['C'], node, parents

    var = C[i, i] if not p else C[i, i] - C[i, p] @ numba_inv(C[np.ix_(p, p)]) @ C[p, i]
    log_prob = -.5*self.n*(1 + np.log2(var/self.n))
    penalty_term = self.lambda_*len(1 + parents)

    return log_prob - penalty_term

