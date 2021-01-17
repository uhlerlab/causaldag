import sys

sys.path.insert(1, "C:/Users/skarn/OneDrive/Documents/MIT/year_3/SuperUROP/causaldag")

from causaldag.utils.scores.monte_carlo_marginal_likelihood import monte_carlo_local_marginal_likelihood, monte_carlo_global_marginal_likelihood
from causaldag.utils.scores.gaussian_bic_score import local_gaussian_bic_score
from functools import partial
import numpy as np
import numba
import scipy as sp
from scipy import stats
from scipy.special import logsumexp
import math
import ipdb
from scipy import stats
from scipy.linalg import ldl

def sample_bge_prior(graph,
        total_num_variables=None,
        inverse_scale_matrix=None,
        degrees_freedom=None,
        alpha_mu=None,
        mu0=None,
        size = 1
):
    p = total_num_variables
    # scale_matrix = np.copy(inverse_scale_matrix)
    # np.fill_diagonal(scale_matrix, 1/np.diagonal(inverse_scale_matrix))
    sigma = stats.invwishart(df = degrees_freedom, scale = inverse_scale_matrix).rvs(size=1, random_state=np.random.default_rng())
    # sigma = np.linalg.inv(inverse_sigma)
    # print(inverse_sigma)
    # print(sigma)
    mu_covariance = (1/alpha_mu) * sigma
    mu = np.random.multivariate_normal(mean = mu0, cov = mu_covariance)
    # print(mu)
    # print(degrees_freedom)
    if size == 1:
        return sigma, mu

    return [sample_bge_prior(graph=graph, total_num_variables=total_num_variables, inverse_scale_matrix=inverse_scale_matrix, degrees_freedom=degrees_freedom, alpha_mu=alpha_mu, mu0=mu0) for _ in range(size)]

def compute_monte_carlo_bge(suffstat, normal_wishart_priors, num_iterations = 1000):
    samples = suffstat["samples"]
    lls = np.zeros(num_iterations)
    for j, (sigma, mu) in enumerate(normal_wishart_priors):
        a = stats.multivariate_normal(mean=mu, cov=sigma).logpdf(samples)
        ll = np.sum(a)
        lls[j] = ll
    
    return logsumexp(lls) - np.log(num_iterations)

if __name__ == '__main__':
    import causaldag
    from causaldag.rand import rand_weights, directed_erdos
    from causaldag.utils.ci_tests import partial_monte_carlo_correlation_suffstat, partial_correlation_suffstat
    from causaldag.utils.scores.gaussian_bge_score import local_gaussian_bge_score
    import time
    # d = causaldag.DAG(arcs={(0, 1)})
    # # d = causaldag.DAG(arcs={(0, 1), (1, 2), (0, 2)})
    # d = causaldag.DAG(arcs={(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)})
    d = causaldag.DAG(arcs={(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)})
    g = rand_weights(d)
    samples = g.sample(100)
    # with open("tests/data/bge_data/samples.npy", 'wb') as f:
    #     np.save(f, samples)
    # samples = np.load("tests/data/bge_data/samples.npy")
    # print(np.shape(samples))
    # Topologically sort data
    print(d.to_amat()[0])
    suffstat = partial_correlation_suffstat(samples)
    suffstat["samples"] = samples
    p = np.shape(samples)[1]
    alpha_mu = p
    alpha_w = p + alpha_mu + 1
    inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    parameter_mean = np.zeros(p)
    num_iterations = 1000
    normal_wishart_priors = sample_bge_prior(d, 
                            total_num_variables=p, 
                            inverse_scale_matrix=inverse_scale_matrix, 
                            degrees_freedom=alpha_w, 
                            alpha_mu=alpha_mu, 
                            mu0=parameter_mean,
                            size=num_iterations)
    sampled_score = compute_monte_carlo_bge(suffstat, normal_wishart_priors, num_iterations)
    print("Sampled BGe Score: ", sampled_score)
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
    
    print("BGe Formula Score: ", total_score_original)
