import numpy as np
import numba
from scipy.special import loggamma
import math
import ipdb

@numba.jit
def numba_inv(A):
    return np.linalg.inv(A)


def local_gaussian_bge_score(
        node,
        parents,
        suffstat,
        alpha_mu=None,
        alpha_w=None,
        inverse_scale_matrix=None,
        parameter_mean=None,
        is_diagonal=True
):
    """
    Compute the BGE score of a node given its parents.

    Parameters
    ----------
    node:
        TODO - describe.
    parents:
        TODO - describe.
    suffstat:
        dictionary containing:

        * ``n`` -- number of samples
        * ``S`` -- sample covariance matrix
        * ``mu`` -- sample mean
    alpha_mu:
        TODO - describe. Default is the number of variables.
    alpha_w:
        TODO - describe. Default is the (number of variables) + alpha_mu + 1
    inverse_scale_matrix:
        TODO - describe. Default is the identity matrix.
    parameter_mean:
        TODO - describe. Default is the zero vector.
    is_diagonal:
        TODO - describe.

    Returns
    -------
    float
        BGE score.
    """
    if not is_diagonal:
        raise NotImplementedError("BGE score not implemented for non-diagonal matrix.")

    k = len(parents)
    n = suffstat["n"]  # number of samples
    S = suffstat["S"]  # sample covariance matrix
    sample_mean = suffstat["mu"]
    p = S.shape[0]  # number of variables

    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)
    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1

    # === first term
    first_term = .5 * np.log(alpha_mu / (n + alpha_mu))

    # === second term: ratio of gamma functions
    second_term = loggamma((n + alpha_w - p + k + 1)/2)
    second_term -= loggamma((alpha_w - p + k + 1)/2)
    second_term -= n/2 * np.log(math.pi)

    # === third term: ratio of determinants
    mean_diff = parameter_mean - sample_mean
    R = inverse_scale_matrix + (n-1) * S + (n * alpha_w) / (n + alpha_w) * np.outer(mean_diff, mean_diff)
    Q = [node, *parents]
    P = list(parents)
    third_term = (alpha_w - p + k + 1)/2 * np.sum(np.log(np.diagonal(inverse_scale_matrix)[Q]))
    third_term += (n + alpha_w - p + k)/2 * np.log(np.linalg.det(R[np.ix_(P, P)]))
    third_term -= (alpha_w - p + k)/2 * np.sum(np.log(np.diagonal(inverse_scale_matrix)[P]))
    third_term -= (n + alpha_w - p + k + 1)/2 * np.log(np.linalg.det(R[np.ix_(Q, Q)]))

    return first_term + second_term + third_term


if __name__ == '__main__':
    from causaldag.rand import rand_weights, directed_erdos
    from causaldag.utils.ci_tests import partial_correlation_suffstat

    d = directed_erdos(10, .5)
    g = rand_weights(d)
    samples = g.sample(100)
    suffstat = partial_correlation_suffstat(samples)
    score = local_gaussian_bge_score(5, d.parents_of(5), suffstat)
