import numpy as np
from scipy import stats
from numpy import log, outer, pi
from numpy.linalg import det


def numba_inv(A):
    return np.linalg.inv(A)


def gaussian_log_likelihood(suffstat, mean, precision):
    sample_cov = suffstat["S"]
    sample_mean = suffstat["mu"]
    nsamples = suffstat["n"]
    p = sample_cov.shape[0]

    constant_term = - p * nsamples * log(2 * pi)
    log_prec_term = nsamples * log(det(precision))
    c = (nsamples - 1) * sample_cov + nsamples * outer(sample_mean - mean, sample_mean - mean)  # TODO: could add to suffstat
    data_term = -np.sum(c * precision)
    ll = .5 * (constant_term + log_prec_term + data_term)

    return ll


if __name__ == '__main__':
    from conditional_independence import partial_correlation_suffstat
    from line_profiler import LineProfiler

    mean = np.zeros(10)
    cov = np.eye(10)
    samples = np.random.multivariate_normal(mean, cov, size=100)

    suffstat = partial_correlation_suffstat(samples)
    suffstat["samples"] = samples

    lp = LineProfiler()
    lp.add_function(gaussian_log_likelihood)

    def run():
        for _ in range(10):
            a = np.random.normal(size=(20, 10))
            gaussian_log_likelihood(suffstat, np.ones(10), a.T @ a)


    lp.runcall(run)
    lp.print_stats()

