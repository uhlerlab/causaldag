import sys

sys.path.insert(1, "C:/Users/skarn/OneDrive/Documents/MIT/year_3/SuperUROP/causaldag")

from causaldag.utils.scores.monte_carlo_marginal_likelihood import monte_carlo_local_marginal_likelihood, monte_carlo_global_marginal_likelihood
from causaldag.utils.scores import gaussian_log_likelihood
from functools import partial
import numpy as np
import numba
import scipy as sp
from scipy import stats
from scipy.special import loggamma
import math
import ipdb
from scipy import stats
from tqdm import tqdm
from scipy.linalg import ldl


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


def local_bge_prior(
        node,
        parents,
        total_num_variables=None,
        inverse_scale_matrix=None,
        degrees_freedom=None,
        alpha_mu=None,
        mu0=None,
        size=1
):
    p = total_num_variables
    variables = [*parents, node]
    k = len(variables)
    incidence = get_complete_dag(k)
    B = np.zeros((k, k))
    indices = np.where(incidence == 1)
    V = list(variables)

    # Normal distribution vectorized function
    standard_normal = lambda t: np.random.normal(0, 1)
    vfunc_standard_normal = np.vectorize(standard_normal)
    if len(B[indices]) > 0:
        B[indices] = vfunc_standard_normal(B[indices])

    c_squared = np.zeros(k)
    for i in range(k):
        c_squared[i] = stats.chi2.rvs(df=degrees_freedom - p + i + 1)

    c = np.sqrt(c_squared)
    inverse_c = 1 / c
    B = np.multiply(-np.array(B), inverse_c)
    scale_matrix = faster_inverse(inverse_scale_matrix)
    scale_matrix_sub_matrix = scale_matrix[variables, variables]
    d = np.zeros((k, k))
    np.fill_diagonal(d, np.multiply(scale_matrix_sub_matrix, c_squared))

    I = np.eye(len(variables))
    A = I - B.T
    inverse_sigma = A.T @ d @ A
    sigma = faster_inverse(inverse_sigma)
    mu_covariance = (1 / alpha_mu) * sigma
    mu = chol_sample(mu0[V], mu_covariance)

    if size == 1:
        return inverse_sigma, B, mu
    else:
        return [
            local_bge_prior(
                node,
                parents,
                total_num_variables=total_num_variables,
                inverse_scale_matrix=inverse_scale_matrix,
                degrees_freedom=degrees_freedom,
                alpha_mu=alpha_mu,
                mu0=mu0
            )
            for _ in range(size)
        ]


def local_gaussian_monte_carlo_bge_score2(
        node,
        parents,
        suffstat: dict,
        alpha_mu=None,
        alpha_w=None,
        inverse_scale_matrix=None,
        parameter_mean=None,
        num_iterations=1000
):
    p = suffstat["C"].shape[0]

    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    bge_prior_partial = partial(
        local_bge_prior,
        total_num_variables=p,
        inverse_scale_matrix=inverse_scale_matrix,
        degrees_freedom=alpha_w,
        alpha_mu=alpha_mu,
        mu0=parameter_mean
    )
    local_score = monte_carlo_local_marginal_likelihood(
        bge_prior_partial,
        local_gaussian_likelihood,
        num_monte_carlo=num_iterations
    )

    return local_score(
        node,
        parents,
        suffstat
    )


def local_gaussian_likelihood(node, parents, suffstat: dict, parameters_list):
    sample_cov = suffstat["C"][np.ix_([*parents, node], [*parents, node])]
    sample_var_node = sample_cov[-1, -1]
    sample_mean = suffstat["mu"][node]
    nsamples = suffstat["n"]

    lls = np.empty(len(parameters_list))
    for j, (precision, B, mu) in enumerate(parameters_list):
        # OLD
        # ll = -.5 * nsamples * np.sum(precision * sample_cov) + .5 * np.log(np.linalg.det(precision)) - .5 * np.log(2 * np.pi)
        # lls[j] = ll

        bias = mu[-1] - B[:-1, -1] @ mu[:-1]
        conditional_precision = precision[-1, -1] - precision[-1, :-1] @ np.linalg.inv(precision[:-1, :-1]) @ precision[:-1, -1]
        constant_term = - .5 * nsamples * np.log(2 * np.pi)
        log_prec_term = .5 * np.log(conditional_precision)
        data_term = -.5 * nsamples * conditional_precision * sample_var_node - .5 * (sample_mean - bias)**2
        lls[j] = constant_term + log_prec_term + data_term
    return lls


def global_bge_prior(
        graph,
        total_num_variables=None,
        inverse_scale_matrix=None,
        degrees_freedom=None,
        alpha_mu=None,
        mu0=None,
        size=1,
        progress=False
):
    p = total_num_variables
    variables = graph.nodes
    k = len(variables)
    B = np.zeros((k, k))
    V = list(variables)
    scale_matrix = faster_inverse(inverse_scale_matrix)

    # # Normal distribution vectorized function
    # standard_normal = lambda t: np.random.normal(0, 1)
    # vfunc_standard_normal = np.vectorize(standard_normal)
    # if len(B[indices]) > 0:
    #     B[indices] = vfunc_standard_normal(B[indices])
    #
    # c_squared = np.zeros(k)
    # for i in range(k):
    #     c_squared[i] = stats.chi2.rvs(df=degrees_freedom - p + i + 1)
    #
    # c = np.sqrt(c_squared)
    # inverse_c = 1 / c
    # B = np.multiply(-np.array(B), inverse_c)
    # d = np.multiply(scale_matrix, c_squared)
    #
    # I = np.eye(len(variables))
    # A = I - B.T
    # inverse_sigma = A.T @ d @ A

    # TODO pull directly from wishart, compare to other way
    inverse_sigmas = stats.wishart(df=degrees_freedom, scale=scale_matrix).rvs(size=size)
    sigmas = [faster_inverse(inverse_sigma) for inverse_sigma in inverse_sigmas]
    # ipdb.set_trace()
    mu_covariances = [(1 / alpha_mu) * sigma for sigma in sigmas]
    # mu = stats.multivariate_normal(mean=mu0[V], cov=mu_covariance).rvs()
    mus = [chol_sample(mu0[V], mu_covariance) for mu_covariance in mu_covariances]

    return list(zip(inverse_sigmas, mus))


def global_monte_carlo_bge_score(
        graph,
        suffstat: dict,
        alpha_mu=None,
        alpha_w=None,
        inverse_scale_matrix=None,
        parameter_mean=None,
        num_iterations=1000,
        progress=False
):
    p = suffstat["C"].shape[0]

    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    bge_prior_partial = partial(
        global_bge_prior,
        total_num_variables=p,
        inverse_scale_matrix=inverse_scale_matrix,
        degrees_freedom=alpha_w,
        alpha_mu=alpha_mu,
        mu0=parameter_mean
    )
    score = monte_carlo_global_marginal_likelihood(
        bge_prior_partial,
        global_gaussian_likelihood,
        num_monte_carlo=num_iterations,
        progress=progress
    )

    return score(
        graph,
        suffstat
    )


def global_gaussian_likelihood(
        graph,
        suffstat: dict,
        parameters_list,
        progress=False
):
    if progress:
        lls = list(tqdm((
            gaussian_log_likelihood(suffstat, mu, precision)
            for precision, mu in parameters_list
        ), total=len(parameters_list)))
    else:
        lls = [
            gaussian_log_likelihood(suffstat, mu, precision)
            for precision, mu in parameters_list
        ]
    return np.array(lls)


if __name__ == '__main__':
    import causaldag
    from causaldag.rand import rand_weights, directed_erdos
    from causaldag.utils.ci_tests import partial_monte_carlo_correlation_suffstat, partial_correlation_suffstat
    from causaldag.utils.scores.gaussian_bge_score import local_gaussian_bge_score
    import time
    # d = causaldag.DAG(arcs={(0, 1)})
    # d = causaldag.DAG(arcs={(0, 1), (1, 2), (0, 2)})
    d = causaldag.DAG(arcs={(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)})
    g = rand_weights(d)
    samples = g.sample(10)
    print(np.shape(samples))
    # Topologically sort data
    print(d.to_amat()[0])
    suffstat = partial_correlation_suffstat(samples)
    suffstat["samples"] = samples
    with open("tests/data/bge_data/samples.npy", 'wb') as f:
        np.save(f, samples)
    with open("tests/data/bge_data/dag_amat", 'wb') as f:
        np.save(f, d.to_amat()[0])
    p = np.shape(samples)[1]
    alpha_mu = p
    alpha_w = p + alpha_mu + 1
    inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    # inverse_scale_matrix = np.eye(p)
    parameter_mean = np.zeros(p)
    s = global_monte_carlo_bge_score(
        d,
        suffstat,
        alpha_mu=alpha_mu,
        alpha_w=alpha_w,
        inverse_scale_matrix=inverse_scale_matrix,
        parameter_mean=parameter_mean,
        num_iterations=100000,
        progress=True
    )
    print(s)

    total_score_original = 0
    for node in range(p):
        total_score_original += local_gaussian_bge_score(
            node,
            d.parents_of(node),
            suffstat,
            alpha_mu=alpha_mu,
            alpha_w=alpha_w,
            inverse_scale_matrix=inverse_scale_matrix,
            parameter_mean=parameter_mean,
        )
    print(total_score_original)
