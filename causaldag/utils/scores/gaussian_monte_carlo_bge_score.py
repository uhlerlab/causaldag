import numpy as np
import numba
import scipy as sp
from scipy import stats
from scipy.special import loggamma
import math
import ipdb
import sys

sys.path.insert(1, "C:/Users/skarn/OneDrive/Documents/MIT/year_3/SuperUROP/causaldag")


@numba.jit
def numba_inv(A):
    return np.linalg.inv(A)


def faster_inverse(A):
    n = A.shape[0]
    b = np.eye(n)
    _, _, x, _ = sp.linalg.lapack.dgesv(A, b, 0, 0)

    return x


def chol_sample(mean, cov):
    return mean + np.linalg.cholesky(cov) @ np.random.standard_normal(mean.size)


def get_complete_dag(n):
    dag_incidence = np.ones((n, n))
    return np.triu(dag_incidence, 1)

def bge_prior(total_num_vars, variables, incidence, inverse_scale_matrix, degrees_freedom):	
    p = total_num_vars	
    k = len(variables)	
    B = np.zeros((k, k))	
    indices = np.where(incidence == 1)	

    # Normal distribution vectorized function	
    standard_normal = lambda t: np.random.normal(0, 1)	
    vfunc_standard_normal = np.vectorize(standard_normal)	
    if len(B[indices]) > 0:	
        B[indices] = vfunc_standard_normal(B[indices])	

    c_squared = np.zeros(k)	
    for i in range(k):	
        c_squared[i] = stats.chi2.rvs(df=degrees_freedom - p + i + 1)	

    c = np.sqrt(c_squared)	
    inverse_c = 1/c	
    B = np.multiply(-np.array(B), inverse_c)	
    scale_matrix = faster_inverse(inverse_scale_matrix)	
    scale_matrix_sub_matrix = scale_matrix[variables, variables]	
    d = np.zeros((k, k))	
    np.fill_diagonal(d, np.multiply(scale_matrix_sub_matrix, c_squared))	

    return B, d

def var_set_monte_carlo_bge_score(
        variables,
        incidence,
        samples,
        alpha_mu=None,
        alpha_w=None,
        inverse_scale_matrix=None,
        parameter_mean=None,
        is_diagonal=True,
        num_iterations=100
):
    if not is_diagonal:
        raise NotImplementedError("BGE score not implemented for non-diagonal matrix.")

    _, p = np.shape(samples)
    num_vars_monte_carlo = len(variables)
    V = list(variables)
    print(V)
    I = np.eye(num_vars_monte_carlo)

    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    df = alpha_w
    parameter_mean_monte_carlo = parameter_mean[V]

    def monte_carlo_iteration(iteration):
        if len(V) == 0:
            return 0
        B, d = bge_prior(p, variables, incidence, inverse_scale_matrix, df)
        B = np.zeros((num_vars_monte_carlo, num_vars_monte_carlo))
        # Compute from formula
        A = I - B.T
        inverse_sigma = A.T @ d @ A
        sigma = faster_inverse(inverse_sigma)
        mu_covariance = (1 / alpha_mu) * sigma
        mu = chol_sample(parameter_mean_monte_carlo, mu_covariance)
        dist = stats.multivariate_normal(parameter_mean_monte_carlo, faster_inverse(d))
        monte_carlo_samples = samples[:, V]
        monte_carlo_sample_margins = monte_carlo_samples - mu
        x_epsilons = (monte_carlo_sample_margins.T - np.dot(B, monte_carlo_sample_margins.T)).T
        prob_x = dist.logpdf(np.array(x_epsilons))

        return np.sum(prob_x)

    log_marginal_likes = [monte_carlo_iteration(i) for i in range(num_iterations)]
    log_marginal_likes_logsumexp = (sp.special.logsumexp(log_marginal_likes) - np.log(num_iterations))

    return log_marginal_likes_logsumexp


def local_gaussian_monte_carlo_bge_score(
        node,
        parents,
        suffstat,
        alpha_mu=None,
        alpha_w=None,
        inverse_scale_matrix=None,
        parameter_mean=None,
        is_diagonal=True,
        num_iterations=1000
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
        TODO - describe.
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

    samples = suffstat['samples']
    samples = np.array(samples)
    k = len(parents)
    _, p = np.shape(samples)

    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    ### First, compute for numerator p(d^{Pa_i U {X_i}} | m^h) ###
    list_parents_and_node = [*parents, node]
    print("list_parents_and_node", list_parents_and_node)
    marginal_likelihood_parents_and_node = var_set_monte_carlo_bge_score(list_parents_and_node, get_complete_dag(k + 1),
                                                                         samples, alpha_mu, alpha_w,
                                                                         inverse_scale_matrix, parameter_mean,
                                                                         is_diagonal, num_iterations)
    # print("marginal_likelihood_parents_and_node", marginal_likelihood_parents_and_node)

    ### First, compute for numerator p(d^{Pa_i | m^h) ###
    list_parents = list(parents)
    marginal_likelihood_parents = var_set_monte_carlo_bge_score(list_parents, get_complete_dag(k), samples, alpha_mu,
                                                                alpha_w, inverse_scale_matrix, parameter_mean,
                                                                is_diagonal, num_iterations)
    # print("marginal_likelihood_parents", marginal_likelihood_parents)
    return (marginal_likelihood_parents_and_node - marginal_likelihood_parents)




    # print("marginal_likelihood_parents", marginal_likelihood_parents)


if __name__ == '__main__':
    import causaldag
    from causaldag.rand import rand_weights, directed_erdos
    from conditional_independence import partial_correlation_suffstat
    from causaldag.utils.ci_tests import partial_monte_carlo_correlation_suffstat, partial_correlation_suffstat
    from causaldag.utils.scores.gaussian_bge_score import local_gaussian_bge_score
    import time

    d = directed_erdos(10, .5)
    g = rand_weights(d)
    ordering = g.topological_sort()
    samples = g.sample(100)
    print(np.shape(samples))
    # Topologically sort data
    samples = samples[:, ordering]
    print(ordering)
    suffstat = partial_monte_carlo_correlation_suffstat(samples)
    node = 7
    # Reorder query and other nodes
    topological_ordering_map = {ordering[i]: i for i in range(len(ordering))}
    ordered_node = topological_ordering_map[node]
    ordered_node_parents = sorted([topological_ordering_map[i] for i in d.parents_of(node)])
    print(d.parents_of(node))
    t = time.process_time()
    score = local_gaussian_monte_carlo_bge_score(ordered_node, ordered_node_parents, suffstat)
    elapsed_time = time.process_time() - t
    print("Elapsed Time: ", elapsed_time)
    score_original = local_gaussian_bge_score(
            ordered_node,
            ordered_node_parents,
            suffstat)
    print("Monte Carlo BGe Score: ", score)
    print("Formula BGe Score: ", score_original)

