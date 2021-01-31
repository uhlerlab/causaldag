import numpy as np
import scipy as sp
from scipy.special import loggamma
import math
import ipdb
import sys

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

    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    # === first term
    first_term = .5 * np.log(alpha_mu / (n + alpha_mu))

    # === second term: ratio of gamma functions
    second_term = loggamma((n + alpha_w - p + k + 1)/2)
    second_term -= loggamma((alpha_w - p + k + 1)/2)
    second_term -= n/2 * np.log(math.pi)

    # === third term: ratio of determinants
    mean_diff = parameter_mean - sample_mean
    # R = inverse_scale_matrix + (n-1) * S + (n * alpha_w) / (n + alpha_w) * np.outer(mean_diff, mean_diff)
    R = inverse_scale_matrix + (n-1) * S + (n * alpha_mu) / (n + alpha_mu) * np.outer(mean_diff, mean_diff)
    # print(R)
    Q = [node, *parents]
    P = list(parents)
    third_term = (alpha_w - p + k + 1)/2 * np.sum(np.log(np.diagonal(inverse_scale_matrix)[Q]))
    third_term += (n + alpha_w - p + k)/2 * np.log(np.linalg.det(R[np.ix_(P, P)]))
    third_term -= (alpha_w - p + k)/2 * np.sum(np.log(np.diagonal(inverse_scale_matrix)[P]))
    third_term -= (n + alpha_w - p + k + 1)/2 * np.log(np.linalg.det(R[np.ix_(Q, Q)]))

    return first_term + second_term + third_term

def partial_correlation_suffstat(samples, invert=True):
    """
    Return the sufficient statistics for partial correlation testing.

    Parameters
    ----------
    samples:
        (n x p) matrix, where n is the number of samples and p is the number of variables.
    invert:
        if True, compute the inverse correlation matrix, and normalize it into the partial correlation matrix. This
        will generally speed up the gauss_ci_test if large conditioning sets are used.

    Returns
    -------
    dict
        dictionary of sufficient statistics
    """
    n, p = samples.shape
    S = np.cov(samples, rowvar=False)  # sample covariance matrix
    mu = np.mean(samples, axis=0)
    # TODO: NaN when variable is deterministic. Replace w/ 1 and 0?
    C = np.corrcoef(samples, rowvar=False)  # sample correlation matrix
    V = samples.T @ samples
    if invert:
        K = np.linalg.pinv(C)
        P = np.linalg.pinv(S)  # sample precision (inverse covariance) matrix
        rho = K/np.sqrt(np.diag(K))/np.sqrt(np.diag(K))[:, None]  # sample partial correlation matrix
        return dict(P=P, S=S, C=C, n=n, K=K, rho=rho, mu=mu)
    return dict(S=S, C=C, n=n, mu=mu, V=V)

if __name__ == '__main__':
    import causaldag
    from causaldag.rand import rand_weights, directed_erdos
    from causaldag.utils.ci_tests import partial_correlation_suffstat, partial_monte_carlo_correlation_suffstat
    from conditional_independence import partial_correlation_suffstat
    from sympy import gamma
    from scipy import stats
    from causaldag.utils.scores.gaussian_monte_carlo_bge_score import local_gaussian_monte_carlo_bge_score

    # d = directed_erdos(10, .5)
    # g = rand_weights(d)
    # samples = g.sample(100)
    # suffstat = partial_correlation_suffstat(samples)
    # score = local_gaussian_bge_score(5, d.parents_of(5), suffstat)

    def integral_marginal_gaussian_bge(data):
        def log_c(a, b):
            c_value = 0
            for i in range(a):
                c_value += loggamma((b - i)/2)

            return c_value

        N_mu = 3
        n = 3
        m = len(data)
        mu_0 = [0, 0, 0]
        T_0 = np.eye(3) * 2.25
        N_w = 7
        # T_0 = np.array([[2.25,-0.512,-0.512], [-0.512,2.25,0.512], [0.512,0.512,2.25]])
        sample_variance = np.sum([np.dot(np.transpose([x - np.average(data, 0)]), [x - np.average(data, 0)]) for x in data], 0)
        print("sample variance", sample_variance)
        T_m = T_0 + sample_variance + ((N_mu*m)/(N_mu + m)) * np.outer((np.array(mu_0) - np.average(data, 0)), (np.array(mu_0) - np.average(data, 0)))
        d_x1 = [0]
        d_x2 = [0, 1]
        d_x3 = [0, 1, 2]
        ell_1 = len(d_x1)
        ell_2 = len(d_x2)
        ell_3 = len(d_x3)
        bge_coefficient_1 = np.log(np.pi) * (-ell_1*m/2) + np.log(N_mu/(N_mu + m)) * (ell_1/2) + log_c(ell_1, m + N_w - n + ell_1) - log_c(ell_1, N_w - n + ell_1)
        bge_coefficient_2 = np.log(np.pi) * (-ell_2*m/2) + np.log(N_mu/(N_mu + m)) * (ell_2/2) + log_c(ell_2, m + N_w - n + ell_2) - log_c(ell_2, N_w - n + ell_2)
        bge_coefficient_3 = np.log(np.pi) * (-ell_3*m/2) + np.log(N_mu/(N_mu + m)) * (ell_3/2) + log_c(ell_3, m + N_w - n + ell_3) - log_c(ell_3, N_w - n + ell_3)
        bge_d_x1 = bge_coefficient_1 + np.log(np.abs(np.linalg.det(T_0[d_x1, :][:, d_x1]))) * ((N_w - n + ell_1)/2) + np.log(np.abs(np.linalg.det(T_m[d_x1, :][:, d_x1]))) * ((-N_w-m+n-ell_1)/2)
        bge_d_x2 = bge_coefficient_2 + np.log(np.abs(np.linalg.det(T_0[d_x2, :][:, d_x2]))) * ((N_w - n + ell_2)/2) + np.log(np.abs(np.linalg.det(T_m[d_x2, :][:, d_x2]))) * ((-N_w-m+n-ell_2)/2)
        bge_d_x3 = bge_coefficient_3 + np.log(np.abs(np.linalg.det(T_0[d_x3, :][:, d_x3]))) * ((N_w - n + ell_3)/2) + np.log(np.abs(np.linalg.det(T_m[d_x3, :][:, d_x3]))) * ((-N_w-m+n-ell_3)/2)

        marginal_1 = bge_d_x1
        marginal_2 = bge_d_x2 - bge_d_x1
        marginal_3 = bge_d_x3 - bge_d_x2
        print("old marginals", marginal_1, marginal_2, marginal_3)
        print("old total", marginal_1 + marginal_2 + marginal_3)

        total_log_marginal_likelihood = bge_coefficient_3 + np.log(np.abs(np.linalg.det(T_0))) * (N_w/2) + np.log(np.abs(np.linalg.det(T_m))) * ((-N_w-m)/2)

        return total_log_marginal_likelihood

    gaussian_data = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
    print("old result", integral_marginal_gaussian_bge(gaussian_data))
    suffstat = partial_monte_carlo_correlation_suffstat(gaussian_data, invert=False)
    s1 = local_gaussian_monte_carlo_bge_score(0, set(), suffstat)
    s2 = local_gaussian_monte_carlo_bge_score(1, {0}, suffstat)
    s3 = local_gaussian_monte_carlo_bge_score(2, {0, 1}, suffstat)
    print("new result node 0", s1)
    print("new result node 1", s2)
    print("new result node 2", s3)
    print("total:", s1+s2+s3)

