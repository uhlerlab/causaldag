import numpy as np
import numba
import scipy as sp
from scipy.special import loggamma
import math
import ipdb
import sys
sys.path.insert(1, "C:/Users/skarn/OneDrive/Documents/MIT/year_3/SuperUROP/causaldag")

from causaldag.utils.ci_tests import partial_monte_carlo_correlation_suffstat, partial_correlation_suffstat
from causaldag.utils.scores.gaussian_bge_score import local_gaussian_bge_score

@numba.jit
def numba_inv(A):
    return np.linalg.inv(A)

def check_ibge_formula_derivation(a, b, a_prime, b_prime, mu_beta, mu_beta_prime, M_beta, M_beta_prime, d_i, d_parents_i):
    print("shape M_beta", np.shape(M_beta))
    n_i, k = np.shape(d_parents_i)
    beta = np.ones(k) * 10
    variance = 1
    
    ### First compute NIG(a, b)
    inverse_gamma_nig_numerator = stats.gamma.logpdf(variance, a=a, scale=1/b)
    normal_nig_numerator = stats.multivariate_normal.logpdf(beta, mean=mu_beta, cov=M_beta)
    
    ### Then compute NIG(a', b')
    inverse_gamma_nig_denominator = stats.gamma.logpdf(1, a=a_prime, scale=1/b_prime)
    covariance = variance * M_beta_prime
    normal_nig_denominator = stats.multivariate_normal.logpdf(beta, mean=mu_beta_prime, cov=covariance)
    normal_numerator = stats.multivariate_normal.logpdf(d_i, mean=d_parents_i @ beta.T, cov=variance * np.eye(n_i))
    
    result = (inverse_gamma_nig_numerator + normal_nig_numerator + normal_numerator) - (inverse_gamma_nig_denominator + normal_nig_denominator)

    return result

def local_bayesian_regression_bge_score(
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
    P = list(parents)
    Q = P + [node]
    n = suffstat["n"]  # number of samples
    sample_mean = suffstat["mu"]
    V = suffstat["V"]
    p = V.shape[0]  # number of variables

    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    scale_matrix = np.multiply(np.eye(p), 1/np.diagonal(inverse_scale_matrix))
    n_i = n
    d_parents_sample_variance = np.zeros((k+1, k+1))
    d_parents_sample_variance[:-1, :-1] = V[np.ix_(P, P)]
    d_parents_sample_variance[:-1, -1] = n_i * sample_mean[P]
    d_parents_sample_variance[-1, :-1] = n_i * sample_mean[P]
    d_parents_sample_variance[-1, -1] = n_i

    mu_beta = np.zeros(k+1)
    M_beta = scale_matrix[np.ix_(Q, Q)]
    M_beta[-1, -1] = 1/alpha_mu
    inverse_M_beta = np.linalg.inv(M_beta)
    
    inverse_M_beta_prime = inverse_M_beta + d_parents_sample_variance
    R = np.linalg.inv(inverse_M_beta_prime)
    d_pa_i_dot_d_i = np.zeros(k+1)
    d_pa_i_dot_d_i[:k] = V[P, node] 
    d_pa_i_dot_d_i[k] = n_i*sample_mean[node]
    mu_beta_prime = R @ (inverse_M_beta @ mu_beta + d_pa_i_dot_d_i)

    # Gamma function terms
    a = (alpha_w - p + k + 1)/2
    b = 1/2 * inverse_scale_matrix[node, node]
    a_prime = a + n_i/2
    b_prime = b + 1/2 * (V[node, node] + mu_beta.T @ inverse_M_beta @ mu_beta - mu_beta_prime.T @ inverse_M_beta_prime @ mu_beta_prime)

    first_term = -np.log(2*np.pi) * n_i/2
    second_term = loggamma(a_prime) - loggamma(a)
    third_term = np.log(b) * a - np.log(b_prime) * a_prime
    _, logdet_M_beta_prime = np.linalg.slogdet(R) 
    _, logdet_M_beta = np.linalg.slogdet(M_beta)
    fourth_term = (logdet_M_beta_prime - logdet_M_beta) * 1/2
    
    # derivation = check_ibge_formula_derivation(a, b, a_prime, b_prime, mu_beta, mu_beta_prime, M_beta, R, d_i, d_parents_i)
    # print("derivation result:", derivation)

    return first_term + second_term + third_term + fourth_term

def local_gaussian_interventional_bge_score(
    node, 
    parents, 
    suffstat_dict, 
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
    P = list(parents)
    V = np.zeros((k+1, k+1))
    sample_mean = np.zeros(k+1)
    n_i = 0
    Q = P + [node]
    p = None
    for intervened_nodes, suffstat in suffstat_dict.items():
        if node not in intervened_nodes:
            if p is None:
                p = suffstat["V"].shape[0]
            V += suffstat["V"][np.ix_(Q, Q)]
            n = suffstat["n"]
            sample_mean += suffstat["mu"][Q] * n
            n_i += n

    if n_i == 0:
        return 0
    
    sample_mean /= n_i

    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    scale_matrix = np.multiply(np.eye(p), 1/np.diagonal(inverse_scale_matrix))
    d_parents_sample_variance = np.zeros((k+1, k+1))
    d_parents_sample_variance[:-1, :-1] = V[:-1, :-1]
    d_parents_sample_variance[:-1, -1] = n_i * sample_mean[:-1]
    d_parents_sample_variance[-1, :-1] = n_i * sample_mean[:-1]
    d_parents_sample_variance[-1, -1] = n_i

    mu_beta = np.zeros(k+1)
    M_beta = scale_matrix[np.ix_(Q, Q)]
    M_beta[-1, -1] = 1/alpha_mu
    inverse_M_beta = np.linalg.inv(M_beta)
    inverse_M_beta_prime = inverse_M_beta + d_parents_sample_variance
    R = np.linalg.inv(inverse_M_beta_prime)
    d_pa_i_dot_d_i = np.zeros(k+1)
    d_pa_i_dot_d_i[:k] = V[:-1, -1] 
    d_pa_i_dot_d_i[k] = n_i*sample_mean[-1]
    mu_beta_prime = R @ (inverse_M_beta @ mu_beta + d_pa_i_dot_d_i)

    # Gamma function terms
    a = (alpha_w - p + k + 1)/2
    b = 1/2 * inverse_scale_matrix[node, node]
    a_prime = a + n_i/2
    b_prime = b + 1/2 * (V[-1, -1] + mu_beta.T @ inverse_M_beta @ mu_beta - mu_beta_prime.T @ inverse_M_beta_prime @ mu_beta_prime)

    first_term = -np.log(2*np.pi) * n_i/2
    second_term = loggamma(a_prime) - loggamma(a)
    third_term = np.log(b) * a - np.log(b_prime) * a_prime
    _, logdet_M_beta_prime = np.linalg.slogdet(R) 
    _, logdet_M_beta = np.linalg.slogdet(M_beta)
    fourth_term = (logdet_M_beta_prime - logdet_M_beta) * 1/2

    return first_term + second_term + third_term + fourth_term

if __name__ == '__main__':
    import causaldag
    from causaldag.rand import rand_weights, directed_erdos
    from causaldag.utils.ci_tests import partial_correlation_suffstat, partial_monte_carlo_correlation_suffstat
    from sympy import gamma
    from scipy import stats
    from causaldag.utils.scores.gaussian_bic_score import local_gaussian_bic_score
    from causaldag.utils.scores.gaussian_monte_carlo_bge_score import local_gaussian_monte_carlo_bge_score

    # d = causaldag.DAG(arcs={(0, 1)})
    # d = causaldag.DAG(arcs={(0, 1), (0, 2), (1, 2)}
    # d = causaldag.DAG(arcs={(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)})
    d = causaldag.DAG(arcs={(i, j) for j in range(10) for i in range(j)})
    g = rand_weights(d)
    samples = g.sample(100)
    # samples = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
    suffstat = partial_correlation_suffstat(samples)
    node = 8
    parents = d.parents_of(node)
    bge_formula_score = local_gaussian_bge_score(node, parents, suffstat)
    ibge_formula_score = local_bayesian_regression_bge_score(node, parents, suffstat)
    bic_formula_score = local_gaussian_bic_score(node, parents, suffstat)
    print("bge_formula_score:  ", bge_formula_score)
    print("ibge_formula_score: ", ibge_formula_score)
    print("bic_formula_score:  ", bic_formula_score)
    print(bge_formula_score - ibge_formula_score)