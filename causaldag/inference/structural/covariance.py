from numpy import sqrt, log1p, ndenumerate, errstate
from scipy.special import erf


def covariance_graph_gauss(ci_tester):
    """
    Estimate a covariance graph by testing whether each pair of nodes is independent, which reduces to
    thresholding correlations (after the Fisher z-transform) for multivariate Gaussian data.

    Parameters
    ----------
    ci_tester:
        Conditional independence tester.

    Examples
    --------
    TODO
    """
    corr = ci_tester.suffstat['C']

    n = ci_tester.suffstat['n']

    # note: log1p(2r/(1-r)) = log((1+r)/(1-r)) but is more numerically stable for r near 0
    # r = 1 causes warnings but gives the correct answer
    with errstate(divide='ignore', invalid='ignore'):
        statistic = sqrt(n - 3) * abs(.5 * log1p(2*corr/(1 - corr)))

    p_values = 1 - .5*(1 + erf(statistic/sqrt(2)))
    alpha = ci_tester.kwargs.get('alpha')
    alpha = alpha if alpha is not None else 1e-5
    edges = {(i, j) for (i, j), p in ndenumerate(p_values) if p < alpha if i != j}


