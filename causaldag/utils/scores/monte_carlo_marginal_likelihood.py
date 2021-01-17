import numpy as np
from scipy.special import logsumexp


def monte_carlo_local_marginal_likelihood(
        prior,
        local_log_likelihood,
        num_monte_carlo=1000
):
    def local_score(node, parents, suffstat):
        parameters_list = prior(node, parents, size=num_monte_carlo)
        lls = local_log_likelihood(node, parents, suffstat, parameters_list)
        return logsumexp(lls) - np.log(num_monte_carlo)

    return local_score


def monte_carlo_global_marginal_likelihood(
        prior,
        log_likelihood,
        num_monte_carlo=1000,
        progress=False
):
    def score(graph, suffstat):
        parameters_list = prior(graph, size=num_monte_carlo, progress=progress)
        lls = log_likelihood(graph, suffstat, parameters_list, progress=progress)
        return logsumexp(lls) - np.log(num_monte_carlo)

    return score
