import numpy as np
import numba

@numba.jit
def numba_inv(A):
    return np.linalg.inv(A)


def local_gaussian_bge_score(
        node,
        parents,
        suffstat,
        alpha_mu,
        inverse_scale_matrix,
        alpha_w
):
    n = suffstat["n"]
    k = len(parents)

    first_term = .5 * np.log(alpha_mu / (n + alpha_mu))
