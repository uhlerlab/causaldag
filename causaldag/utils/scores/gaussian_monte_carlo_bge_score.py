import numpy as np
import numba
import scipy as sp
from scipy import stats
from scipy.special import loggamma
import math
import ipdb

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
    I = np.eye(num_vars_monte_carlo)

    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    scale_matrix = faster_inverse(inverse_scale_matrix)
    df = alpha_w
    parameter_mean_monte_carlo = parameter_mean[V]
    scale_matrix_monte_carlo = scale_matrix[V, :][:, V]
    if num_iterations == 0 or len(variables) == 0:
        return 0
    
    standard_normal = lambda t: np.random.normal(0, 1)
    vfunc_standard_normal = np.vectorize(standard_normal)
    c_squared = np.zeros((num_iterations, num_vars_monte_carlo))
    
    for var in range(num_vars_monte_carlo):
        c_squared[:, var] = stats.chi2.rvs(df - p + var + 1, size = num_iterations)
        
    c = np.sqrt(c_squared)
    indices = np.where(incidence == 1)
    inverse_c = 1/np.array(c)

    def monte_carlo_iteration(iteration):
        B = np.zeros((num_vars_monte_carlo, num_vars_monte_carlo))
        
        if len(B[indices]) > 0:
            B[indices] = vfunc_standard_normal(B[indices])
        
        B = np.multiply(-np.array(B), inverse_c[iteration])
        d = np.zeros((num_vars_monte_carlo, num_vars_monte_carlo))
        np.fill_diagonal(d, scale_matrix_monte_carlo @ c_squared[iteration])
        # Compute from formula
        A = I - B.T
        inverse_sigma = A.T @ d @ A
        sigma = faster_inverse(inverse_sigma)
        mu_covariance = (1/alpha_mu) * sigma
        mu = chol_sample(parameter_mean_monte_carlo, mu_covariance) 
        log_likelihood_sum = 0
        dist = stats.multivariate_normal(parameter_mean_monte_carlo, faster_inverse(d))
        monte_carlo_samples = samples[:, V]
    
        for data_point in monte_carlo_samples:
            x_epsilon = (data_point - mu) - np.dot(B, data_point - mu)
            prob_x = dist.logpdf(x_epsilon)
            log_likelihood_sum += prob_x
        
        return log_likelihood_sum

    log_marginal_likes = list(map(monte_carlo_iteration, range(num_iterations)))
    log_marginal_likes_logsumexp = (sp.special.logsumexp(log_marginal_likes) - np.log(num_iterations)) 

    return log_marginal_likes_logsumexp

def local_gaussian_monte_carlo_bge_score(
        node,
        parents,
        samples,
        alpha_mu=None,
        alpha_w=None,
        inverse_scale_matrix=None,
        parameter_mean=None,
        is_diagonal=True,
        num_iterations=100
):
    """
    Compute the BGE score of a node given its parents.

    Parameters
    ----------
    node:
        TODO - describe.
    parents:
        TODO - describe.
    samples:
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
    marginal_likelihood_parents_and_node = var_set_monte_carlo_bge_score(list_parents_and_node, get_complete_dag(k+1), samples, alpha_mu, alpha_w, inverse_scale_matrix, parameter_mean, is_diagonal, num_iterations)
    # print("marginal_likelihood_parents_and_node", marginal_likelihood_parents_and_node)
    
    ### First, compute for numerator p(d^{Pa_i | m^h) ###
    list_parents = list(parents)
    marginal_likelihood_parents = var_set_monte_carlo_bge_score(list_parents, get_complete_dag(k), samples, alpha_mu, alpha_w, inverse_scale_matrix, parameter_mean, is_diagonal, num_iterations)
    # print("marginal_likelihood_parents", marginal_likelihood_parents)
    return (marginal_likelihood_parents_and_node - marginal_likelihood_parents)

if __name__ == '__main__':
    gaussian_data = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
    s1 = local_gaussian_monte_carlo_bge_score(0, set(), gaussian_data)
    s2 = local_gaussian_monte_carlo_bge_score(1, {0}, gaussian_data)
    s3 = local_gaussian_monte_carlo_bge_score(2, {0, 1}, gaussian_data)
    print("new result node 0", s1)
    print("new result node 1", s2)
    print("new result node 2", s3)
    print("total:", s1+s2+s3)

